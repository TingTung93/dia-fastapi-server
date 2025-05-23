"""
FastAPI Server for Dia TTS Model - SillyTavern Compatible
"""

import io
import os
import re
import tempfile
import time
import json
import uuid
import threading
import asyncio
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from queue import Queue as ThreadQueue
from multiprocessing import Queue as MPQueue, Process
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Header

from dia import Dia


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=4096, description="Text to convert to speech")
    voice_id: str = Field(default="alloy", description="Voice identifier")
    response_format: str = Field(default="wav", description="Audio format (wav, mp3)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    role: Optional[str] = Field(default=None, description="Role of the speaker (user, assistant, system)")
    
    # Dia model parameters
    temperature: Optional[float] = Field(default=None, ge=0.1, le=2.0, description="Sampling temperature")
    cfg_scale: Optional[float] = Field(default=None, ge=1.0, le=10.0, description="Classifier-free guidance scale")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    max_tokens: Optional[int] = Field(default=None, ge=100, le=10000, description="Maximum tokens to generate")
    use_torch_compile: Optional[bool] = Field(default=None, description="Enable torch.compile optimization")


class VoiceInfo(BaseModel):
    name: str
    voice_id: str
    preview_url: Optional[str] = None


class TTSGenerateRequest(BaseModel):
    """Legacy format for backwards compatibility"""
    text: str = Field(..., max_length=4096)
    voice_id: str = Field(default="alloy")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class VoiceMapping(BaseModel):
    style: str
    primary_speaker: str
    audio_prompt: Optional[str] = None
    audio_prompt_transcript: Optional[str] = None


class VoiceMappingUpdate(BaseModel):
    voice_id: str
    style: Optional[str] = None
    primary_speaker: Optional[str] = None
    audio_prompt: Optional[str] = None
    audio_prompt_transcript: Optional[str] = None


class ServerConfig(BaseModel):
    debug_mode: bool = False
    save_outputs: bool = False
    show_prompts: bool = False
    output_retention_hours: int = 24


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TTSJob(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    text: str
    processed_text: Optional[str] = None
    voice_id: str
    speed: float
    role: Optional[str] = None
    temperature: Optional[float] = None
    cfg_scale: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    use_torch_compile: Optional[bool] = None
    audio_prompt_used: bool = False
    generation_time: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None


class GenerationLog(BaseModel):
    id: str
    timestamp: datetime
    text: str
    processed_text: str
    voice: str
    audio_prompt_used: bool
    generation_time: float
    file_path: Optional[str] = None
    file_size: Optional[int] = None


class QueueStats(BaseModel):
    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_workers: int
    active_workers: int


# Initialize FastAPI app
app = FastAPI(
    title="Dia TTS Server",
    description="FastAPI server for Dia text-to-speech model, compatible with SillyTavern",
    version="1.0.0"
)

# Add CORS middleware for web compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (single GPU mode)
model: Optional[Dia] = None

# Multi-GPU support
GPU_MODELS: Dict[int, Dia] = {}  # GPU ID -> Model instance
GPU_WORKERS: Dict[int, Process] = {}  # GPU ID -> Worker process
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

# GPU mode configuration
GPU_MODE = os.getenv("DIA_GPU_MODE", "auto").lower()
GPU_IDS_STR = os.getenv("DIA_GPU_IDS", "")
if GPU_IDS_STR:
    ALLOWED_GPUS = [int(gpu_id.strip()) for gpu_id in GPU_IDS_STR.split(",")]
    GPU_COUNT = len(ALLOWED_GPUS)
else:
    ALLOWED_GPUS = list(range(GPU_COUNT))

# Determine if we use multi-GPU
if GPU_MODE == "multi":
    USE_MULTI_GPU = GPU_COUNT > 1
elif GPU_MODE == "single":
    USE_MULTI_GPU = False
else:  # auto
    USE_MULTI_GPU = GPU_COUNT > 1

# Security (optional, accepts any bearer token)
security = HTTPBearer(auto_error=False)

# Voice mapping (Dia uses speaker tags [S1]/[S2], we'll map common voice names)
VOICE_MAPPING: Dict[str, Dict[str, Any]] = {
    "alloy": {"style": "neutral", "primary_speaker": "S1", "audio_prompt": None, "audio_prompt_transcript": None},
    "echo": {"style": "calm", "primary_speaker": "S1", "audio_prompt": None, "audio_prompt_transcript": None}, 
    "fable": {"style": "expressive", "primary_speaker": "S2", "audio_prompt": None, "audio_prompt_transcript": None},
    "nova": {"style": "friendly", "primary_speaker": "S1", "audio_prompt": None, "audio_prompt_transcript": None},
    "onyx": {"style": "deep", "primary_speaker": "S2", "audio_prompt": None, "audio_prompt_transcript": None},
    "shimmer": {"style": "bright", "primary_speaker": "S1", "audio_prompt": None, "audio_prompt_transcript": None},
}

# Store uploaded audio prompts (now stores file paths)
AUDIO_PROMPTS: Dict[str, str] = {}
AUDIO_PROMPT_DIR = "audio_prompts"

# Server configuration
SERVER_CONFIG = ServerConfig()

# Generation logs
GENERATION_LOGS: Dict[str, GenerationLog] = {}

# Job queue and management
JOB_QUEUE: Dict[str, TTSJob] = {}
JOB_RESULTS: Dict[str, bytes] = {}  # Store audio results in memory

# Output directory for saved files
OUTPUT_DIR = "audio_outputs"

# Rich console for pretty output
console = Console()

# Worker management
WORKER_POOL: Optional[ThreadPoolExecutor] = None
# For multi-GPU: 1 worker per GPU, for single GPU: configurable via env
DEFAULT_WORKERS = GPU_COUNT if USE_MULTI_GPU else min(4, mp.cpu_count())
MAX_WORKERS = int(os.getenv("DIA_MAX_WORKERS", DEFAULT_WORKERS))
ACTIVE_WORKERS: Dict[str, bool] = {}

# GPU assignment for workers (round-robin)
WORKER_GPU_ASSIGNMENT: Dict[int, int] = {}  # Worker ID -> GPU ID
NEXT_GPU = 0  # For round-robin assignment
NEXT_SYNC_GPU = 0  # For synchronous requests


def load_model():
    """Load the Dia model on startup"""
    global model, GPU_MODELS, USE_MULTI_GPU
    
    if model is not None or GPU_MODELS:
        return
    
    console.print(Panel(
        f"[bold cyan]Loading Dia TTS Model[/bold cyan]\n\n"
        f"Detected GPUs: {GPU_COUNT}\n"
        f"GPU Mode: {GPU_MODE}\n"
        f"Worker Count: {MAX_WORKERS}",
        title="[bold]Model Initialization[/bold]",
        border_style="cyan"
    ))
    
    if USE_MULTI_GPU and GPU_COUNT > 1:
        console.print(f"[cyan]Using multi-GPU mode with {GPU_COUNT} GPUs[/cyan]")
        console.print(f"[cyan]Setting worker count to {MAX_WORKERS} (1 per GPU)[/cyan]")
        load_multi_gpu_models()
    else:
        load_single_model()


def load_single_model():
    """Load a single model instance (single GPU or CPU mode)"""
    global model
    
    try:
        # Determine device
        if torch.cuda.is_available():
            # Use first allowed GPU if specified
            if ALLOWED_GPUS:
                device = torch.device(f"cuda:{ALLOWED_GPUS[0]}")
                console.print(f"[yellow]Using GPU {ALLOWED_GPUS[0]} for single model mode[/yellow]")
            else:
                device = torch.device("cuda")
            # Use bfloat16 if available (faster based on benchmarks), fall back to float16
            if torch.cuda.is_bf16_supported():
                compute_dtype = "bfloat16"
                console.print("[green]Using BFloat16 precision for better performance[/green]")
            else:
                compute_dtype = "float16"
                console.print("[yellow]BFloat16 not supported, using Float16[/yellow]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps") 
            compute_dtype = "float16"
        else:
            device = torch.device("cpu")
            compute_dtype = "float32"
        
        console.print(f"[cyan]Device: {device}, Precision: {compute_dtype}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Loading Dia model from Hugging Face...[/cyan]", total=100)
            
            # Simulate progress during download
            progress.update(task, advance=30)
            
            model = Dia.from_pretrained(
                "nari-labs/Dia-1.6B", 
                compute_dtype=compute_dtype,
                device=device
            )
            
            progress.update(task, completed=100)
        
        console.print("[bold green]✓ Dia model loaded successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error loading Dia model: {e}[/bold red]")
        raise RuntimeError(f"Failed to load Dia model: {e}")


def load_multi_gpu_models():
    """Load one model instance per GPU"""
    global GPU_MODELS
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        main_task = progress.add_task(
            f"[cyan]Loading models on {len(ALLOWED_GPUS)} GPUs...[/cyan]", 
            total=len(ALLOWED_GPUS)
        )
        
        for gpu_id in ALLOWED_GPUS:
            try:
                device = torch.device(f"cuda:{gpu_id}")
                
                # Check compute capability for this specific GPU
                with torch.cuda.device(gpu_id):
                    if torch.cuda.is_bf16_supported():
                        compute_dtype = "bfloat16"
                        console.print(f"[green]GPU {gpu_id}: Using BFloat16 precision[/green]")
                    else:
                        compute_dtype = "float16"
                        console.print(f"[yellow]GPU {gpu_id}: Using Float16 precision[/yellow]")
                
                console.print(f"[cyan]Loading model on GPU {gpu_id}...[/cyan]")
                
                GPU_MODELS[gpu_id] = Dia.from_pretrained(
                    "nari-labs/Dia-1.6B",
                    compute_dtype=compute_dtype,
                    device=device
                )
                
                console.print(f"[green]✓ Model loaded successfully on GPU {gpu_id}[/green]")
                progress.advance(main_task)
                
            except Exception as e:
                console.print(f"[red]✗ Error loading model on GPU {gpu_id}: {e}[/red]")
                # Continue loading on other GPUs
    
    if not GPU_MODELS:
        raise RuntimeError("Failed to load model on any GPU")
    
    console.print(f"[bold green]✓ Successfully loaded {len(GPU_MODELS)} model instances across GPUs[/bold green]")


def preprocess_text(text: str, voice_id: str, role: Optional[str] = None) -> str:
    """Preprocess text for Dia model requirements"""
    # Remove asterisks (common in chat applications)
    text = re.sub(r'\*+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure text has proper speaker tags for Dia
    if not ('[S1]' in text or '[S2]' in text):
        # Determine speaker based on role if provided
        if role:
            # Map roles to speakers: user -> S1, assistant/system -> S2
            if role.lower() == "user":
                primary_speaker = "S1"
            elif role.lower() in ["assistant", "system"]:
                primary_speaker = "S2"
            else:
                # Unknown role, fall back to voice mapping
                voice_config = VOICE_MAPPING.get(voice_id, VOICE_MAPPING["alloy"])
                primary_speaker = voice_config["primary_speaker"]
        else:
            # No role provided, use voice mapping to determine primary speaker
            voice_config = VOICE_MAPPING.get(voice_id, VOICE_MAPPING["alloy"])
            primary_speaker = voice_config["primary_speaker"]
        
        # Wrap text with proper closing tags: [S1] text [S1]
        text = f"[{primary_speaker}] {text} [{primary_speaker}]"
    else:
        # Ensure existing tags are properly closed
        # Simple approach: if we find an opening tag without a closing tag, add it
        if '[S1]' in text and not text.endswith('[S1]'):
            if not text.endswith('[S2]'):
                text += ' [S1]'
        elif '[S2]' in text and not text.endswith('[S2]'):
            if not text.endswith('[S1]'):
                text += ' [S2]'
    
    return text


def can_use_torch_compile() -> bool:
    """Check if torch.compile can be used safely"""
    # Check if disabled via environment variable
    if os.getenv("DIA_DISABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes"):
        return False
    
    try:
        # Only try torch.compile on CUDA with proper compiler setup
        if not torch.cuda.is_available():
            return False
        
        # Check if we're on Windows and don't have proper compiler
        import platform
        if platform.system() == "Windows":
            # On Windows, torch.compile often fails without proper MSVC setup
            return False
        
        # Try a simple compilation test
        @torch.compile
        def test_fn(x):
            return x + 1
        
        test_tensor = torch.tensor([1.0])
        test_fn(test_tensor)
        return True
    except Exception:
        return False


def ensure_output_dir():
    """Ensure output directory exists"""
    if SERVER_CONFIG.save_outputs and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_audio_prompt_dir():
    """Ensure audio prompt directory exists"""
    if not os.path.exists(AUDIO_PROMPT_DIR):
        os.makedirs(AUDIO_PROMPT_DIR, exist_ok=True)


def cleanup_old_files():
    """Remove files older than retention period"""
    if not SERVER_CONFIG.save_outputs or not os.path.exists(OUTPUT_DIR):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=SERVER_CONFIG.output_retention_hours)
    
    # Clean up files
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Failed to clean up {filename}: {e}")
    
    # Clean up logs for deleted files
    logs_to_remove = []
    for log_id, log in GENERATION_LOGS.items():
        if log.file_path and not os.path.exists(log.file_path):
            logs_to_remove.append(log_id)
    
    for log_id in logs_to_remove:
        del GENERATION_LOGS[log_id]


def process_tts_job(job_id: str) -> None:
    """Process a TTS job in worker thread"""
    worker_id = f"worker_{threading.current_thread().ident}"
    ACTIVE_WORKERS[worker_id] = True
    
    try:
        job = JOB_QUEUE.get(job_id)
        if not job:
            return
        
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        job.worker_id = worker_id
        
        if SERVER_CONFIG.debug_mode:
            console.print(f"\n[bold yellow]>>> Job Started[/bold yellow]")
            console.print(f"[cyan]Job ID:[/cyan] {job_id[:8]}...")
            console.print(f"[cyan]Worker:[/cyan] {worker_id}")
            console.print(f"[cyan]Voice:[/cyan] {job.voice_id}")
        
        # Generate audio
        audio_data, log_id = generate_audio_from_text(
            job.text,
            job.voice_id,
            job.speed,
            job.temperature,
            job.cfg_scale,
            job.top_p,
            job.max_tokens,
            job.use_torch_compile,
            job.role,
            worker_id=worker_id
        )
        
        # Convert to bytes for storage
        with io.BytesIO() as buffer:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            sf.write(buffer, audio_data, 44100, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_bytes = buffer.getvalue()
        
        # Store result
        JOB_RESULTS[job_id] = audio_bytes
        
        # Update job
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.generation_time = (job.completed_at - job.started_at).total_seconds()
        
        if SERVER_CONFIG.debug_mode:
            console.print(f"[bold green]>>> Job Completed[/bold green]")
            console.print(f"[cyan]Job ID:[/cyan] {job_id[:8]}...")
            console.print(f"[cyan]Time:[/cyan] {job.generation_time:.2f}s")
        
    except Exception as e:
        job = JOB_QUEUE.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
        
        if SERVER_CONFIG.debug_mode:
            console.print(f"[bold red]>>> Job Failed[/bold red]")
            console.print(f"[cyan]Job ID:[/cyan] {job_id[:8]}...")
            console.print(f"[red]Error:[/red] {e}")
    
    finally:
        ACTIVE_WORKERS[worker_id] = False


def get_model_for_worker(worker_id: int = 0):
    """Get the appropriate model instance for a worker"""
    if USE_MULTI_GPU and GPU_MODELS:
        # Assign worker to GPU in round-robin fashion
        gpu_idx = worker_id % len(ALLOWED_GPUS)
        gpu_id = ALLOWED_GPUS[gpu_idx]
        return GPU_MODELS.get(gpu_id), gpu_id
    else:
        # Single model mode
        return model, 0


def create_generation_progress(text: str, voice_id: str, gpu_id: int, worker_id: int):
    """Create a rich progress display for generation"""
    # Create progress table
    table = Table(title=f"[bold cyan]TTS Generation - Worker {worker_id} (GPU {gpu_id})[/bold cyan]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    # Truncate text for display
    display_text = text[:100] + "..." if len(text) > 100 else text
    
    table.add_row("Text", display_text)
    table.add_row("Voice", voice_id)
    table.add_row("GPU", f"cuda:{gpu_id}")
    table.add_row("Worker", str(worker_id))
    
    return table


def generate_audio_from_text(
    text: str, 
    voice_id: str = "alloy", 
    speed: float = 0.8,
    temperature: Optional[float] = None,
    cfg_scale: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_torch_compile: Optional[bool] = None,
    role: Optional[str] = None,
    worker_id: int = 0
) -> tuple[np.ndarray, str]:
    """Generate audio using Dia model and return (audio, log_id)"""
    # Get model for this worker
    model_instance, gpu_id = get_model_for_worker(worker_id)
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    start_time = time.time()
    log_id = str(uuid.uuid4())
    
    # Preprocess text
    processed_text = preprocess_text(text, voice_id, role)
    
    # Get voice configuration
    voice_config = VOICE_MAPPING.get(voice_id, VOICE_MAPPING["alloy"])
    
    # Add audio prompt transcript if available (for better voice cloning)
    if voice_config.get("audio_prompt_transcript") and voice_config.get("audio_prompt"):
        # Prepend the transcript for better voice cloning results
        processed_text = voice_config["audio_prompt_transcript"] + " " + processed_text
    
    # Get audio prompt if available
    audio_prompt = None
    audio_prompt_used = False
    if voice_config.get("audio_prompt"):
        audio_prompt_path = AUDIO_PROMPTS.get(voice_config["audio_prompt"])
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            # Model expects file path for audio prompt
            audio_prompt = audio_prompt_path
            audio_prompt_used = True
        else:
            if audio_prompt_path:
                print(f"Warning: Audio prompt file not found: {audio_prompt_path}")
            audio_prompt_used = False
    
    # Set default parameters
    generation_params = {
        "temperature": temperature or 1.2,
        "cfg_scale": cfg_scale or 3.0,
        "top_p": top_p or 0.95,
        "max_tokens": max_tokens,
        "use_torch_compile": use_torch_compile if use_torch_compile is not None else can_use_torch_compile(),
        "verbose": SERVER_CONFIG.debug_mode
    }
    
    # Create progress display
    if SERVER_CONFIG.debug_mode or SERVER_CONFIG.show_prompts:
        info_table = create_generation_progress(text, voice_id, gpu_id, worker_id)
        
        # Add additional info
        panel = Panel(
            info_table,
            expand=False,
            border_style="cyan"
        )
        console.print(panel)
        
        # Show generation parameters
        param_text = Text()
        param_text.append("Generation Parameters:\n", style="bold yellow")
        param_text.append(f"  Temperature: {generation_params['temperature']}\n")
        param_text.append(f"  CFG Scale: {generation_params['cfg_scale']}\n")
        param_text.append(f"  Top-p: {generation_params['top_p']}\n")
        if generation_params['max_tokens']:
            param_text.append(f"  Max Tokens: {generation_params['max_tokens']}\n")
        param_text.append(f"  Audio Prompt: {'Yes' if audio_prompt_used else 'No'}\n")
        param_text.append(f"  Processed Text: {processed_text[:100]}...\n" if len(processed_text) > 100 else f"  Processed Text: {processed_text}\n")
        
        console.print(param_text)
    
    # Generate audio
    try:
        # Generate with proper audio prompt handling
        generation_start = time.time()
        
        # Create progress bar for generation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            # Add generation task
            gen_task = progress.add_task(
                f"[cyan]Generating audio on GPU {gpu_id}...[/cyan]",
                total=100
            )
            
            # Update progress periodically (we'll simulate progress since model doesn't provide callbacks)
            def update_progress():
                elapsed = 0
                while elapsed < 30:  # Max 30 seconds timeout
                    time.sleep(0.1)
                    elapsed = time.time() - generation_start
                    # Simulate progress based on typical generation time (3-5 seconds)
                    simulated_progress = min(95, (elapsed / 4.0) * 100)
                    progress.update(gen_task, completed=simulated_progress)
                    if hasattr(update_progress, 'stop'):
                        break
            
            # Start progress updater in background
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            try:
                # Set verbose=True to get token speed info from model
                generation_params['verbose'] = SERVER_CONFIG.debug_mode
                
                audio_output = model_instance.generate(
                    processed_text,
                    audio_prompt=audio_prompt,  # Dia will handle the batching internally
                    **generation_params
                )
                
                # Stop progress thread
                update_progress.stop = True
                progress.update(gen_task, completed=100)
                
            finally:
                update_progress.stop = True
        
        # Apply speed adjustment if needed
        if speed != 1.0 and audio_output is not None:
            # Simple speed adjustment by resampling
            original_len = len(audio_output)
            target_len = int(original_len / speed)
            if target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                audio_output = np.interp(x_resampled, x_original, audio_output)
        
        generation_time = time.time() - start_time
        
        # Calculate performance metrics
        if audio_output is not None and (SERVER_CONFIG.debug_mode or SERVER_CONFIG.show_prompts):
            # Estimate tokens based on audio length (44.1kHz, ~86 tokens/sec)
            audio_duration = len(audio_output) / 44100  # seconds
            estimated_tokens = int(audio_duration * 86)  # 86 tokens/sec from model
            tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0
            realtime_factor = audio_duration / generation_time if generation_time > 0 else 0
            
            # Display performance metrics
            perf_panel = Panel(
                f"[bold green]✓ Generation Complete[/bold green]\n\n"
                f"Time: {generation_time:.2f}s\n"
                f"Audio Duration: {audio_duration:.2f}s\n"
                f"Estimated Tokens: {estimated_tokens}\n"
                f"Speed: {tokens_per_sec:.1f} tokens/sec\n"
                f"Realtime Factor: {realtime_factor:.2f}x",
                title="[bold]Performance Metrics[/bold]",
                border_style="green"
            )
            console.print(perf_panel)
        
        # Save output file if enabled
        file_path = None
        file_size = None
        if SERVER_CONFIG.save_outputs and audio_output is not None:
            ensure_output_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{log_id[:8]}_{voice_id}.wav"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save audio file
            sf.write(file_path, audio_output, 44100, format='WAV', subtype='PCM_16')
            file_size = os.path.getsize(file_path)
        
        # Create log entry
        log_entry = GenerationLog(
            id=log_id,
            timestamp=datetime.now(),
            text=text,
            processed_text=processed_text,
            voice=voice_id,
            audio_prompt_used=audio_prompt_used,
            generation_time=generation_time,
            file_path=file_path,
            file_size=file_size
        )
        GENERATION_LOGS[log_id] = log_entry
        
        if SERVER_CONFIG.debug_mode:
            print(f"Generation completed in {generation_time:.2f}s")
            if file_path:
                print(f"Saved to: {file_path}")
        
        return audio_output, log_id
        
    except Exception as e:
        # If torch.compile fails, try again without it
        if "Compiler:" in str(e) and "not found" in str(e):
            print("Torch compile failed, retrying without compilation...")
            try:
                # Retry with compilation disabled
                retry_params = generation_params.copy()
                retry_params["use_torch_compile"] = False
                
                audio_output = model.generate(
                    processed_text,
                    audio_prompt=audio_prompt,
                    **retry_params
                )
                
                # Apply speed adjustment if needed
                if speed != 1.0 and audio_output is not None:
                    original_len = len(audio_output)
                    target_len = int(original_len / speed)
                    if target_len > 0:
                        x_original = np.arange(original_len)
                        x_resampled = np.linspace(0, original_len - 1, target_len)
                        audio_output = np.interp(x_resampled, x_original, audio_output)
                
                generation_time = time.time() - start_time
                
                # Save and log as above
                file_path = None
                file_size = None
                if SERVER_CONFIG.save_outputs and audio_output is not None:
                    ensure_output_dir()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{log_id[:8]}_{voice_id}.wav"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    sf.write(file_path, audio_output, 44100, format='WAV', subtype='PCM_16')
                    file_size = os.path.getsize(file_path)
                
                log_entry = GenerationLog(
                    id=log_id,
                    timestamp=datetime.now(),
                    text=text,
                    processed_text=processed_text,
                    voice=voice_id,
                    audio_prompt_used=audio_prompt_used,
                    generation_time=generation_time,
                    file_path=file_path,
                    file_size=file_size
                )
                GENERATION_LOGS[log_id] = log_entry
                
                return audio_output, log_id
            except Exception as retry_e:
                print(f"Retry without compilation also failed: {retry_e}")
                raise HTTPException(status_code=500, detail=f"Audio generation failed: {retry_e}")
        
        print(f"Error generating audio: {e}")
        print(f"Text: {processed_text}")
        print(f"Voice: {voice_id}")
        print(f"Audio prompt available: {audio_prompt is not None}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")


def initialize_worker_pool():
    """Initialize the worker thread pool"""
    global WORKER_POOL
    WORKER_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="TTS-Worker")
    
    console.print(Panel(
        f"[bold green]✓ Worker Pool Initialized[/bold green]\n\n"
        f"Total Workers: {MAX_WORKERS}\n"
        f"Mode: {'Multi-GPU' if USE_MULTI_GPU else 'Single-GPU/CPU'}\n"
        f"GPUs per Worker: {1 if USE_MULTI_GPU else 'Shared'}",
        title="[bold]Worker Configuration[/bold]",
        border_style="green"
    ))


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    initialize_worker_pool()
    
    # Ensure audio prompt directory exists
    ensure_audio_prompt_dir()
    
    # Load existing audio prompts from disk
    if os.path.exists(AUDIO_PROMPT_DIR):
        for filename in os.listdir(AUDIO_PROMPT_DIR):
            if filename.endswith('.wav'):
                prompt_id = filename[:-4]  # Remove .wav extension
                file_path = os.path.join(AUDIO_PROMPT_DIR, filename)
                AUDIO_PROMPTS[prompt_id] = file_path
                print(f"Loaded audio prompt: {prompt_id}")
    
    # Start cleanup task
    def cleanup_task():
        while True:
            cleanup_old_files()
            # Clean up old job results (keep for 1 hour)
            cleanup_old_jobs()
            time.sleep(3600)  # Run every hour
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()


def cleanup_old_jobs():
    """Clean up completed jobs and results older than 1 hour"""
    cutoff_time = datetime.now() - timedelta(hours=1)
    
    jobs_to_remove = []
    for job_id, job in JOB_QUEUE.items():
        if job.completed_at and job.completed_at < cutoff_time:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        JOB_QUEUE.pop(job_id, None)
        JOB_RESULTS.pop(job_id, None)
        if SERVER_CONFIG.debug_mode:
            print(f"Cleaned up old job: {job_id}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global WORKER_POOL
    if WORKER_POOL:
        WORKER_POOL.shutdown(wait=True)
        print("Worker pool shut down")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Dia TTS Server is running", "status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": "dia",
                "name": "Dia TTS",
                "description": "1.6B parameter text-to-speech model for dialogue generation"
            }
        ]
    }


@app.get("/voices")
async def list_voices():
    """List available voices"""
    voices = []
    for voice_id, config in VOICE_MAPPING.items():
        voices.append({
            "id": voice_id,
            "name": voice_id,
            "style": config["style"],
            "primary_speaker": config["primary_speaker"],
            "has_audio_prompt": config.get("audio_prompt") is not None,
            "preview_url": f"/preview/{voice_id}"
        })
    return {"voices": voices}


@app.post("/generate")
async def generate_speech(
    request: TTSRequest, 
    async_mode: bool = Query(default=False, description="Return job ID for async processing")
):
    """Main TTS generation endpoint - supports both sync and async modes"""
    # Log incoming request
    if SERVER_CONFIG.debug_mode:
        console.print(f"\n[bold blue]>>> New TTS Request[/bold blue]")
        console.print(f"[cyan]Mode:[/cyan] {'Async' if async_mode else 'Sync'}")
        console.print(f"[cyan]Voice:[/cyan] {request.voice_id}")
        console.print(f"[cyan]Text Length:[/cyan] {len(request.text)} chars")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if async_mode:
        # Async mode: return job ID immediately
        job_id = str(uuid.uuid4())
        job = TTSJob(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            role=request.role,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            use_torch_compile=request.use_torch_compile
        )
        
        JOB_QUEUE[job_id] = job
        
        # Submit to worker pool
        if WORKER_POOL:
            WORKER_POOL.submit(process_tts_job, job_id)
        else:
            raise HTTPException(status_code=503, detail="Worker pool not available")
        
        return {"job_id": job_id, "status": "pending", "message": "Job queued for processing"}
    
    else:
        # Sync mode: traditional immediate response
        try:
            # Get next GPU for sync request (round-robin)
            global NEXT_SYNC_GPU
            sync_worker_id = NEXT_SYNC_GPU
            if USE_MULTI_GPU:
                NEXT_SYNC_GPU = (NEXT_SYNC_GPU + 1) % len(ALLOWED_GPUS)
            
            # Generate audio
            audio_data, log_id = generate_audio_from_text(
                request.text, 
                request.voice_id, 
                request.speed,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                use_torch_compile=request.use_torch_compile,
                role=request.role,
                worker_id=sync_worker_id
            )
            
            if audio_data is None:
                raise HTTPException(status_code=500, detail="Failed to generate audio")
            
            # Convert to bytes and create streaming response
            def generate_audio_stream():
                with io.BytesIO() as buffer:
                    # Ensure audio is in the right format
                    if audio_data.dtype != np.float32:
                        audio_data_processed = audio_data.astype(np.float32)
                    else:
                        audio_data_processed = audio_data
                    
                    # Write audio file
                    sf.write(buffer, audio_data_processed, 44100, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    
                    # Stream in chunks
                    chunk_size = 8192
                    while True:
                        chunk = buffer.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            
            # Determine media type and filename
            if request.response_format.lower() == "mp3":
                media_type = "audio/wav"  # Still return WAV for now
                filename = "speech.wav"
            else:
                media_type = "audio/wav"
                filename = "speech.wav"
            
            response_headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "Transfer-Encoding": "chunked"
            }
            
            # Add log ID header if debug mode
            if SERVER_CONFIG.debug_mode:
                response_headers["X-Generation-ID"] = log_id
            
            return StreamingResponse(
                generate_audio_stream(),
                media_type=media_type,
                headers=response_headers
            )
            
        except Exception as e:
            print(f"Error in generate_speech: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/generate")
async def generate_speech_alt(request: TTSGenerateRequest):
    """Legacy TTS endpoint for backwards compatibility"""
    tts_request = TTSRequest(
        text=request.text,
        voice_id=request.voice_id
    )
    return await generate_speech(tts_request)


@app.get("/api/tts/speakers")
async def list_speakers():
    """List speakers (alternative format)"""
    return list(VOICE_MAPPING.keys())


@app.get("/preview/{voice_id}")
async def get_voice_preview(voice_id: str):
    """Generate a preview sample for a voice"""
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    preview_text = f"[S1] Hello, this is a preview of the {voice_id} voice. [S2] How does this sound to you?"
    
    try:
        audio_data, log_id = generate_audio_from_text(preview_text, voice_id)
        
        with io.BytesIO() as buffer:
            sf.write(buffer, audio_data, 44100, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_bytes = buffer.getvalue()
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=preview_{voice_id}.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }


# Voice Management Endpoints

@app.get("/voice_mappings")
async def get_voice_mappings():
    """Get current voice mappings"""
    return VOICE_MAPPING


@app.put("/voice_mappings/{voice_id}")
async def update_voice_mapping(voice_id: str, update: VoiceMappingUpdate):
    """Update voice mapping configuration"""
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    # Update voice configuration
    if update.style is not None:
        VOICE_MAPPING[voice_id]["style"] = update.style
    if update.primary_speaker is not None:
        VOICE_MAPPING[voice_id]["primary_speaker"] = update.primary_speaker
    if update.audio_prompt is not None:
        VOICE_MAPPING[voice_id]["audio_prompt"] = update.audio_prompt
    if update.audio_prompt_transcript is not None:
        VOICE_MAPPING[voice_id]["audio_prompt_transcript"] = update.audio_prompt_transcript
    
    return {"message": f"Voice '{voice_id}' updated successfully", "voice_config": VOICE_MAPPING[voice_id]}


@app.post("/voice_mappings")
async def create_voice_mapping(mapping: VoiceMappingUpdate):
    """Create new voice mapping"""
    if not mapping.voice_id:
        raise HTTPException(status_code=400, detail="voice_id is required")
    
    VOICE_MAPPING[mapping.voice_id] = {
        "style": mapping.style or "neutral",
        "primary_speaker": mapping.primary_speaker or "S1",
        "audio_prompt": mapping.audio_prompt,
        "audio_prompt_transcript": mapping.audio_prompt_transcript
    }
    
    return {"message": f"Voice '{mapping.voice_id}' created successfully", "voice_config": VOICE_MAPPING[mapping.voice_id]}


@app.delete("/voice_mappings/{voice_id}")
async def delete_voice_mapping(voice_id: str):
    """Delete voice mapping (only custom voices, not defaults)"""
    default_voices = {"alloy", "echo", "fable", "nova", "onyx", "shimmer"}
    
    if voice_id in default_voices:
        raise HTTPException(status_code=400, detail=f"Cannot delete default voice '{voice_id}'")
    
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    # Remove associated audio prompt
    if VOICE_MAPPING[voice_id].get("audio_prompt"):
        audio_prompt_id = VOICE_MAPPING[voice_id]["audio_prompt"]
        AUDIO_PROMPTS.pop(audio_prompt_id, None)
    
    del VOICE_MAPPING[voice_id]
    return {"message": f"Voice '{voice_id}' deleted successfully"}


@app.post("/audio_prompts/upload")
async def upload_audio_prompt(
    prompt_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Upload audio file to use as voice prompt"""
    # Basic validation
    if not prompt_id or not prompt_id.strip():
        raise HTTPException(status_code=400, detail="prompt_id cannot be empty")
    
    if not audio_file or not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Content type validation (more flexible)
    valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    file_ext = os.path.splitext(audio_file.filename.lower())[1]
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {', '.join(valid_extensions)}"
        )
    
    temp_file_path = None
    try:
        # Read audio file data
        audio_data = await audio_file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Create temporary file with proper extension
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=file_ext)
        
        try:
            # Write data to temp file
            with os.fdopen(temp_file_fd, 'wb') as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data is written to disk
            
            # Load audio with error handling
            try:
                audio_array, sample_rate = sf.read(temp_file_path)
            except Exception as sf_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot read audio file. Please check the file format: {str(sf_error)}"
                )
            
            # Validate audio data
            if len(audio_array) == 0:
                raise HTTPException(status_code=400, detail="Audio file contains no audio data")
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Validate duration (3-30 seconds recommended)
            duration = len(audio_array) / sample_rate
            if duration < 0.5:
                raise HTTPException(status_code=400, detail="Audio file too short (minimum 0.5 seconds)")
            if duration > 60:
                raise HTTPException(status_code=400, detail="Audio file too long (maximum 60 seconds)")
            
            # Resample to 44.1kHz if needed
            if sample_rate != 44100:
                resample_ratio = 44100 / sample_rate
                new_length = int(len(audio_array) * resample_ratio)
                if new_length > 0:
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array) - 1, new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    )
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            
            # Save audio file to disk for the model to load
            ensure_audio_prompt_dir()
            audio_prompt_path = os.path.join(AUDIO_PROMPT_DIR, f"{prompt_id}.wav")
            
            # Save as WAV file at 44.1kHz
            sf.write(audio_prompt_path, audio_array, 44100, format='WAV', subtype='PCM_16')
            
            # Store the file path
            AUDIO_PROMPTS[prompt_id] = audio_prompt_path
            
            return {
                "message": f"Audio prompt '{prompt_id}' uploaded successfully",
                "duration": len(audio_array) / 44100,
                "sample_rate": 44100,
                "original_sample_rate": sample_rate,
                "channels": "mono"
            }
            
        except HTTPException:
            raise
        except Exception as process_error:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing audio file: {str(process_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during file upload: {str(e)}"
        )
    finally:
        # Always clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")


@app.get("/audio_prompts")
async def list_audio_prompts():
    """List available audio prompts"""
    prompts = {}
    for prompt_id, file_path in AUDIO_PROMPTS.items():
        if os.path.exists(file_path):
            try:
                # Read file info
                audio_data, sr = sf.read(file_path)
                prompts[prompt_id] = {
                    "file_path": file_path,
                    "duration": len(audio_data) / sr,
                    "sample_rate": sr,
                    "exists": True
                }
            except Exception as e:
                prompts[prompt_id] = {
                    "file_path": file_path,
                    "exists": True,
                    "error": str(e)
                }
        else:
            prompts[prompt_id] = {
                "file_path": file_path,
                "exists": False
            }
    return prompts


@app.delete("/audio_prompts/{prompt_id}")
async def delete_audio_prompt(prompt_id: str):
    """Delete audio prompt"""
    if prompt_id not in AUDIO_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Audio prompt '{prompt_id}' not found")
    
    # Check if any voices are using this prompt
    using_voices = [voice_id for voice_id, config in VOICE_MAPPING.items() 
                   if config.get("audio_prompt") == prompt_id]
    
    if using_voices:
        return {
            "warning": f"Audio prompt '{prompt_id}' is used by voices: {using_voices}",
            "message": "Remove from voice mappings first before deleting"
        }
    
    # Delete the file
    file_path = AUDIO_PROMPTS[prompt_id]
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete audio file {file_path}: {e}")
    
    del AUDIO_PROMPTS[prompt_id]
    return {"message": f"Audio prompt '{prompt_id}' deleted successfully"}


# Debug and Configuration Endpoints

@app.get("/config")
async def get_server_config():
    """Get current server configuration"""
    return SERVER_CONFIG


@app.put("/config")
async def update_server_config(config: ServerConfig):
    """Update server configuration"""
    global SERVER_CONFIG
    SERVER_CONFIG = config
    return {"message": "Configuration updated successfully", "config": SERVER_CONFIG}


@app.get("/logs")
async def get_generation_logs(
    limit: int = Query(default=50, le=500),
    voice: Optional[str] = Query(default=None)
):
    """Get generation logs"""
    logs = list(GENERATION_LOGS.values())
    
    # Filter by voice if specified
    if voice:
        logs = [log for log in logs if log.voice == voice]
    
    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Limit results
    logs = logs[:limit]
    
    return {
        "logs": logs,
        "total": len(GENERATION_LOGS),
        "filtered": len(logs)
    }


@app.get("/logs/{log_id}")
async def get_generation_log(log_id: str):
    """Get specific generation log"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    return GENERATION_LOGS[log_id]


@app.get("/logs/{log_id}/download")
async def download_generation_output(log_id: str):
    """Download the audio file for a specific generation"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    log = GENERATION_LOGS[log_id]
    if not log.file_path or not os.path.exists(log.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or has been cleaned up")
    
    filename = os.path.basename(log.file_path)
    return FileResponse(
        log.file_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.delete("/logs")
async def clear_generation_logs():
    """Clear all generation logs"""
    global GENERATION_LOGS
    GENERATION_LOGS = {}
    return {"message": "All generation logs cleared"}


@app.delete("/logs/{log_id}")
async def delete_generation_log(log_id: str):
    """Delete specific generation log and its file"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    log = GENERATION_LOGS[log_id]
    
    # Delete file if it exists
    if log.file_path and os.path.exists(log.file_path):
        try:
            os.remove(log.file_path)
        except Exception as e:
            print(f"Failed to delete file {log.file_path}: {e}")
    
    # Delete log
    del GENERATION_LOGS[log_id]
    
    return {"message": f"Log '{log_id}' and associated file deleted"}


# Job Management Endpoints

@app.get("/jobs")
async def list_jobs(
    status: Optional[JobStatus] = Query(default=None),
    limit: int = Query(default=50, le=500)
):
    """List jobs with optional status filter"""
    jobs = list(JOB_QUEUE.values())
    
    # Filter by status if specified
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    return {
        "jobs": jobs,
        "total": len(JOB_QUEUE),
        "filtered": len(jobs)
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of specific job"""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return JOB_QUEUE[job_id]


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Download result of completed job"""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = JOB_QUEUE[job_id]
    
    if job.status != JobStatus.COMPLETED:
        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=f"Job failed: {job.error_message}")
        else:
            raise HTTPException(status_code=425, detail=f"Job not completed (status: {job.status})")
    
    if job_id not in JOB_RESULTS:
        raise HTTPException(status_code=404, detail="Job result not found or expired")
    
    audio_bytes = JOB_RESULTS[job_id]
    
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=speech_{job_id[:8]}.wav",
            "Content-Length": str(len(audio_bytes))
        }
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending job"""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = JOB_QUEUE[job_id]
    
    if job.status == JobStatus.PENDING:
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        return {"message": f"Job '{job_id}' cancelled"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job.status}")


@app.get("/queue/stats")
async def get_queue_stats():
    """Get queue statistics"""
    stats = {
        "pending_jobs": len([j for j in JOB_QUEUE.values() if j.status == JobStatus.PENDING]),
        "processing_jobs": len([j for j in JOB_QUEUE.values() if j.status == JobStatus.PROCESSING]),
        "completed_jobs": len([j for j in JOB_QUEUE.values() if j.status == JobStatus.COMPLETED]),
        "failed_jobs": len([j for j in JOB_QUEUE.values() if j.status == JobStatus.FAILED]),
        "total_workers": MAX_WORKERS,
        "active_workers": len([w for w in ACTIVE_WORKERS.values() if w])
    }
    
    return QueueStats(**stats)


@app.get("/gpu/status")
async def get_gpu_status():
    """Get GPU configuration and status"""
    gpu_info = {
        "gpu_mode": GPU_MODE,
        "gpu_count": GPU_COUNT,
        "allowed_gpus": ALLOWED_GPUS,
        "use_multi_gpu": USE_MULTI_GPU,
        "models_loaded": {}
    }
    
    if USE_MULTI_GPU:
        # Show which GPUs have models loaded
        for gpu_id in GPU_MODELS:
            gpu_info["models_loaded"][f"gpu_{gpu_id}"] = True
    else:
        # Single model mode
        gpu_info["models_loaded"]["single_model"] = model is not None
        if model is not None and torch.cuda.is_available():
            gpu_info["single_model_gpu"] = ALLOWED_GPUS[0] if ALLOWED_GPUS else 0
    
    # Get GPU memory info if available
    if torch.cuda.is_available():
        gpu_info["gpu_memory"] = {}
        for gpu_id in ALLOWED_GPUS:
            try:
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                
                gpu_info["gpu_memory"][f"gpu_{gpu_id}"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "free_gb": round(memory_total - memory_reserved, 2)
                }
            except Exception as e:
                gpu_info["gpu_memory"][f"gpu_{gpu_id}"] = {"error": str(e)}
    
    return gpu_info


@app.delete("/jobs")
async def clear_completed_jobs():
    """Clear all completed and failed jobs"""
    jobs_to_remove = []
    for job_id, job in JOB_QUEUE.items():
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        JOB_QUEUE.pop(job_id, None)
        JOB_RESULTS.pop(job_id, None)
    
    return {"message": f"Cleared {len(jobs_to_remove)} completed jobs"}


@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old files and jobs"""
    cleanup_old_files()
    cleanup_old_jobs()
    return {"message": "Cleanup completed"}




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-outputs", action="store_true", help="Save audio outputs to files")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts in console")
    parser.add_argument("--retention-hours", type=int, default=24, help="File retention hours")
    
    args = parser.parse_args()
    
    # Update server config from command line args
    if args.debug:
        SERVER_CONFIG.debug_mode = True
    if args.save_outputs:
        SERVER_CONFIG.save_outputs = True
    if args.show_prompts:
        SERVER_CONFIG.show_prompts = True
    SERVER_CONFIG.output_retention_hours = args.retention_hours
    
    print(f"Starting Dia TTS Server on {args.host}:{args.port}")
    print("Make sure you have set the HF_TOKEN environment variable!")
    print(f"SillyTavern endpoint: http://{args.host}:{args.port}/v1/audio/speech")
    print(f"Configuration API: http://{args.host}:{args.port}/v1/config")
    print(f"Generation logs: http://{args.host}:{args.port}/v1/logs")
    print(f"Debug mode: {SERVER_CONFIG.debug_mode}")
    print(f"Save outputs: {SERVER_CONFIG.save_outputs}")
    print(f"Show prompts: {SERVER_CONFIG.show_prompts}")
    print(f"Retention: {SERVER_CONFIG.output_retention_hours} hours")
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )