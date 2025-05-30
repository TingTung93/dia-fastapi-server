"""
Simplified FastAPI Server for Dia TTS Model - GPU Performance Focused
"""

import io
import os
import time
import numpy as np
import soundfile as sf
import torch
import uvicorn
import json
import datetime
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from dia import Dia
from rich.console import Console
from rich.panel import Panel
import asyncio

# Initialize console for output
console = Console()

# Constants
AUDIO_OUTPUT_DIR = Path("audio_outputs")
AUDIO_RETENTION_HOURS = 24

# Create audio output directory if it doesn't exist
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=4096, description="Text to convert to speech")
    voice_id: str = Field(default="alloy", description="Voice identifier")
    response_format: str = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=100, le=10000)

class OpenAITTSRequest(BaseModel):
    model: str = Field(default="dia")
    input: str = Field(..., max_length=4096)
    voice: str = Field(default="alloy")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

# Initialize FastAPI app
app = FastAPI(
    title="Dia TTS Server (Simplified)",
    description="High-performance GPU-focused TTS server",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = None
compute_dtype = None

# Voice to speaker mapping
VOICE_MAPPINGS = {
    "alloy": {"style": "neutral", "primary_speaker": "S1"},
    "echo": {"style": "whispering", "primary_speaker": "S1"},
    "fable": {"style": "excited", "primary_speaker": "S1"},
    "onyx": {"style": "sad", "primary_speaker": "S1"},
    "nova": {"style": "angry", "primary_speaker": "S1"},
    "shimmer": {"style": "friendly", "primary_speaker": "S1"},
}

# Audio prompts storage
AUDIO_PROMPTS: Dict[str, Any] = {}

def load_model():
    """Load the Dia model on the best available GPU"""
    global model, device, compute_dtype
    
    if model is not None:
        return
    
    console.print("[cyan]Initializing Dia TTS Model...[/cyan]")
    
    # Select best device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # Check for BFloat16 support
        if torch.cuda.is_bf16_supported():
            compute_dtype = "bfloat16"
            console.print("[green]Using BFloat16 precision for optimal performance[/green]")
        else:
            compute_dtype = "float16"
            console.print("[yellow]Using Float16 precision[/yellow]")
            
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"[cyan]GPU: {gpu_name} ({memory_gb:.1f}GB)[/cyan]")
    else:
        device = torch.device("cpu")
        compute_dtype = "float32"
        console.print("[yellow]No GPU detected, using CPU (slower)[/yellow]")
    
    try:
        # Load model
        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B",
            compute_dtype=compute_dtype,
            device=device
        )
        
        # Enable torch.compile for better performance
        if hasattr(torch, 'compile') and device.type == "cuda":
            try:
                console.print("[cyan]Compiling model with torch.compile for better performance...[/cyan]")
                model = torch.compile(model, mode="reduce-overhead")
                console.print("[green]✓ Model compiled successfully[/green]")
            except Exception as e:
                console.print(f"[yellow]torch.compile not available or failed: {e}[/yellow]")
        
        console.print("[bold green]✓ Dia model loaded successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error loading model: {e}[/bold red]")
        raise

def process_text_for_voice(text: str, voice_id: str) -> str:
    """Process text with speaker tags based on voice selection"""
    voice_config = VOICE_MAPPINGS.get(voice_id, VOICE_MAPPINGS["alloy"])
    primary_speaker = voice_config["primary_speaker"]
    style = voice_config["style"]
    
    # Add speaker tags
    processed_text = f"[{primary_speaker}] {text} [{primary_speaker}]"
    
    # Add style prefix
    if style != "neutral":
        processed_text = f"{style}: {processed_text}"
    
    return processed_text

def save_audio_output(audio_data: bytes, metadata: dict) -> str:
    """Save audio output with metadata for debugging"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_id = f"{timestamp}_{metadata['voice_id']}"
    
    # Create output directory
    output_dir = AUDIO_OUTPUT_DIR / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save audio file
    audio_path = output_dir / "audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_data)
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "generation_params": metadata,
            "file_path": str(audio_path)
        }, f, indent=2)
    
    return output_id

async def cleanup_old_outputs():
    """Clean up audio outputs older than retention period"""
    try:
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=AUDIO_RETENTION_HOURS)
        
        for output_dir in AUDIO_OUTPUT_DIR.iterdir():
            if not output_dir.is_dir():
                continue
                
            # Parse directory timestamp from name
            try:
                dir_time = datetime.datetime.strptime(output_dir.name.split("_")[0], "%Y%m%d_%H%M%S")
                if dir_time < cutoff_time:
                    shutil.rmtree(output_dir)
                    console.print(f"[yellow]Cleaned up old output: {output_dir.name}[/yellow]")
            except (ValueError, IndexError):
                continue
    except Exception as e:
        console.print(f"[red]Error during cleanup: {e}[/red]")

@app.on_event("startup")
async def startup_event():
    """Load model and start cleanup task on startup"""
    load_model()
    
    # Start periodic cleanup
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Run cleanup periodically"""
    while True:
        await cleanup_old_outputs()
        await asyncio.sleep(3600)  # Run every hour

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "dia",
        "version": "2.0.0",
        "gpu": torch.cuda.is_available(),
        "device": str(device) if device else "not loaded"
    }

@app.post("/generate")
async def generate_tts(request: TTSRequest) -> Response:
    """Generate TTS audio synchronously"""
    try:
        # Set a reasonable timeout for generation
        timeout = 120  # 2 minutes max
        
        # Generate audio with background task
        audio_data = await asyncio.wait_for(
            asyncio.to_thread(
                generate_audio,
                text=request.text,
                voice_id=request.voice_id,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                speed=request.speed
            ),
            timeout=timeout
        )
        
        # Return audio
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=speech.wav",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "*"
            }
        )
        
    except asyncio.TimeoutError:
        console.print("[red]Generation timed out[/red]")
        raise HTTPException(
            status_code=504,
            detail=f"Generation timed out after {timeout} seconds"
        )
    except Exception as e:
        console.print(f"[red]Generation error: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "dia"}

@app.post("/v1/audio/speech")
async def openai_tts(request: OpenAITTSRequest) -> StreamingResponse:
    """OpenAI-compatible TTS endpoint for SillyTavern"""
    try:
        # Generate audio
        audio_data = generate_audio(
            text=request.input,
            voice_id=request.voice,
            speed=request.speed,
            temperature=1.0,  # Use default values for OpenAI endpoint
            cfg_scale=3.0,
            top_p=0.95,
            max_tokens=2000
        )
        
        # Stream the response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        console.print(f"[red]OpenAI endpoint error: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio_prompts/upload")
async def upload_audio_prompt(
    prompt_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Upload custom audio prompt for voice cloning"""
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Load audio with soundfile
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)
        
        # Resample to 44.1kHz if needed
        if sample_rate != 44100:
            # Simple resampling
            duration = len(audio_array) / sample_rate
            target_samples = int(duration * 44100)
            x_original = np.linspace(0, len(audio_array) - 1, len(audio_array))
            x_resampled = np.linspace(0, len(audio_array) - 1, target_samples)
            audio_array = np.interp(x_resampled, x_original, audio_array)
        
        # Reshape for batch processing if needed
        if len(audio_array.shape) == 1:
            audio_array = audio_array.reshape(1, -1)
        
        # Store audio prompt
        AUDIO_PROMPTS[prompt_id] = audio_array
        
        return {"message": f"Audio prompt '{prompt_id}' uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

@app.get("/voices")
async def list_voices():
    """List available voices"""
    voices = list(VOICE_MAPPINGS.keys()) + list(AUDIO_PROMPTS.keys())
    return {"voices": voices}

@app.get("/gpu/status")
async def gpu_status():
    """Get GPU status and memory usage"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),
        "memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),
        "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
    }

@app.get("/outputs")
async def list_outputs():
    """List available audio outputs"""
    outputs = []
    for output_dir in AUDIO_OUTPUT_DIR.iterdir():
        if not output_dir.is_dir():
            continue
            
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                outputs.append(metadata)
    
    return {"outputs": sorted(outputs, key=lambda x: x["timestamp"], reverse=True)}

@app.get("/outputs/{output_id}")
async def get_output(output_id: str):
    """Get specific audio output"""
    output_dir = AUDIO_OUTPUT_DIR / output_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    
    audio_path = output_dir / "audio.wav"
    metadata_path = output_dir / "metadata.json"
    
    if not all([audio_path.exists(), metadata_path.exists()]):
        raise HTTPException(status_code=404, detail="Output files missing")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        headers={
            "X-Output-Metadata": json.dumps(metadata)
        }
    )

@app.delete("/outputs/{output_id}")
async def delete_output(output_id: str):
    """Delete specific audio output"""
    output_dir = AUDIO_OUTPUT_DIR / output_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    
    shutil.rmtree(output_dir)
    return {"message": f"Output {output_id} deleted"}

def generate_audio(
    text: str,
    voice_id: str = "alloy",
    temperature: float = 1.0,
    cfg_scale: float = 3.0,
    top_p: float = 0.95,
    max_tokens: int = 2000,
    speed: float = 1.0
) -> bytes:
    """Generate audio using the Dia model"""
    start_time = time.time()
    
    try:
        # Process text
        processed_text = process_text_for_voice(text, voice_id)
        
        # Get audio prompt if custom voice
        audio_prompt = None
        if voice_id in AUDIO_PROMPTS:
            audio_prompt = AUDIO_PROMPTS[voice_id]
            # Convert to batch format if needed
            if len(audio_prompt.shape) == 1:
                audio_prompt = audio_prompt.reshape(1, -1)
        
        # Generate audio with CUDA synchronization
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            audio_output = model.generate(
                processed_text,
                audio_prompt=audio_prompt,
                temperature=temperature,
                cfg_scale=cfg_scale,
                top_p=top_p,
                max_tokens=max_tokens,
                verbose=False
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Convert to numpy array if needed
        if isinstance(audio_output, torch.Tensor):
            audio_output = audio_output.cpu().numpy()
        
        # Ensure we have a 1D array
        if len(audio_output.shape) > 1:
            audio_output = audio_output.squeeze()
        
        # Apply speed adjustment if needed
        if speed != 1.0 and audio_output is not None:
            original_len = len(audio_output)
            target_len = int(original_len / speed)
            if target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                audio_output = np.interp(x_resampled, x_original, audio_output)
        
        # Convert to WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_output, 44100, format='WAV', subtype='PCM_16')
        wav_data = wav_buffer.getvalue()
        
        # Save output for debugging
        metadata = {
            "text": text,
            "voice_id": voice_id,
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "speed": speed,
            "generation_time": time.time() - start_time,
            "audio_duration": len(audio_output) / 44100
        }
        output_id = save_audio_output(wav_data, metadata)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        audio_duration = len(audio_output) / 44100
        estimated_tokens = int(audio_duration * 86)  # ~86 tokens/sec
        tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0
        realtime_factor = audio_duration / generation_time if generation_time > 0 else 0
        
        # Display performance metrics
        perf_text = f"""
Generation Time: {generation_time:.2f}s
Audio Duration: {audio_duration:.2f}s
Estimated Tokens: {estimated_tokens}
Speed: {tokens_per_sec:.1f} tokens/sec
Realtime Factor: {realtime_factor:.2f}x
Output ID: {output_id}
        """
        console.print(Panel(perf_text, title="[green]Performance Metrics[/green]", border_style="green"))
        
        return wav_data
        
    except Exception as e:
        console.print(f"[red]Error in generate_audio: {e}[/red]")
        raise

if __name__ == "__main__":
    # Get configuration from environment or defaults
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "7860"))
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )