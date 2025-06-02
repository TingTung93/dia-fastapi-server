"""Pytest configuration and fixtures for Dia TTS server tests."""

import sys
from unittest.mock import Mock, MagicMock, patch

# Mock the dia module before any imports
sys.modules['dia'] = MagicMock()
mock_dia_class = MagicMock()
sys.modules['dia'].Dia = mock_dia_class

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient
from typing import Generator, Any

# Path handling is done by pytest.ini pythonpath setting


@pytest.fixture
def mock_dia_model():
    """Mock Dia model for testing without GPU."""
    model = MagicMock()
    model.generate_audio = MagicMock()
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    
    # Mock audio generation
    def mock_generate(text, **kwargs):
        # Return fake audio data (1 second of silence at 44100 Hz)
        audio = np.zeros((44100,), dtype=np.float32)
        metrics = {
            "tokens_generated": len(text) * 2,
            "generation_time": 0.5,
            "tokens_per_second": len(text) * 4
        }
        return audio, 44100, metrics
    
    model.generate_audio.side_effect = mock_generate
    return model


@pytest.fixture
def mock_torch():
    """Mock torch module for testing without GPU."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.cuda.device_count', return_value=0), \
         patch('torch.compile', lambda x, **kwargs: x):
        yield


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create temporary directories for audio files."""
    audio_outputs = tmp_path / "audio_outputs"
    audio_prompts = tmp_path / "audio_prompts"
    audio_outputs.mkdir()
    audio_prompts.mkdir()
    
    # Create some test audio prompt files
    test_prompts = ["test1.wav", "test2.wav"]
    for prompt in test_prompts:
        # Create dummy WAV file
        (audio_prompts / prompt).write_bytes(b"RIFF....WAVEfmt ")
    
    return {
        "outputs": str(audio_outputs),
        "prompts": str(audio_prompts)
    }


@pytest.fixture
def mock_env_vars(temp_audio_dir):
    """Mock environment variables."""
    env_vars = {
        "HF_TOKEN": "test_token",
        "API_KEY": "test_api_key",
        "AUDIO_OUTPUT_DIR": temp_audio_dir["outputs"],
        "AUDIO_PROMPT_DIR": temp_audio_dir["prompts"],
        "GPU_MODE": "single",
        "WORKERS": "2",
        "PORT": "7860",
        "HOST": "127.0.0.1"
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def sample_text():
    """Sample text for TTS generation."""
    return "Hello, this is a test of the text to speech system."


@pytest.fixture
def test_voice_mappings():
    """Create test voice mappings for testing."""
    return {
        "test_voice": {
            "style": "neutral", 
            "primary_speaker": "S1", 
            "audio_prompt": None, 
            "audio_prompt_transcript": None
        },
        "alloy": {  # Keep for backward compatibility with existing tests
            "style": "neutral", 
            "primary_speaker": "S1", 
            "audio_prompt": None, 
            "audio_prompt_transcript": None
        }
    }

@pytest.fixture
def sample_request_data():
    """Sample request data for API tests."""
    return {
        "text": "Hello, this is a test.",
        "voice_id": "test_voice",
        "response_format": "wav",
        "speed": 1.0
    }


@pytest.fixture
def mock_audio_data():
    """Mock audio data for testing."""
    # 1 second of sine wave at 440 Hz (A4 note)
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32), sample_rate


@pytest.fixture
def mock_job_result(mock_audio_data):
    """Mock job result for testing."""
    audio, sr = mock_audio_data
    return {
        "status": "completed",
        "audio_data": audio,
        "sample_rate": sr,
        "file_path": "/tmp/test_output.wav",
        "generation_time": 0.5,
        "created_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T00:00:01",
        "metadata": {
            "tokens_generated": 100,
            "tokens_per_second": 200.0
        }
    }


@pytest.fixture
def client(mock_torch, mock_dia_model, test_voice_mappings):
    """Test client for FastAPI with all necessary mocks."""
    with patch('src.server.DIA_MODEL', mock_dia_model), \
         patch.dict('src.server.VOICE_MAPPING', test_voice_mappings, clear=True):
        from src.server import app
        return TestClient(app)

@pytest.fixture
async def async_client():
    """Async test client for FastAPI."""
    # This will be used for actual integration tests
    # For unit tests, we'll mock the components
    pass


@pytest.fixture
def mock_worker_pool():
    """Mock worker pool for testing."""
    pool = MagicMock()
    pool.submit = MagicMock()
    pool.shutdown = MagicMock()
    
    # Mock future object
    future = MagicMock()
    future.result = MagicMock(return_value=None)
    pool.submit.return_value = future
    
    return pool


@pytest.fixture(autouse=True)
def mock_server_startup(test_voice_mappings):
    """Automatically mock server startup functions and inject test voices."""
    with patch('src.server.load_model') as mock_load, \
         patch('src.server.initialize_worker_pool') as mock_init, \
         patch.dict('src.server.VOICE_MAPPING', test_voice_mappings, clear=True):
        # Prevent actual initialization
        mock_load.return_value = None
        mock_init.return_value = None
        yield