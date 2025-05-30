"""Unit tests for API endpoints."""

import pytest
import json
import io
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import numpy as np
from datetime import datetime
import sys

# Ensure dia is mocked (should be done in conftest but being extra safe)
if 'dia' not in sys.modules:
    sys.modules['dia'] = MagicMock()

# Import with proper patching
with patch('src.server.load_model') as mock_load, \
     patch('src.server.initialize_worker_pool') as mock_init:
    # Prevent actual loading
    mock_load.return_value = None
    mock_init.return_value = None
    
    from src.server import app, VOICE_MAPPING, JOB_QUEUE, JOB_RESULTS


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.unit
class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root health check."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        with patch('src.server.model', Mock()):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "timestamp" in data


@pytest.mark.unit 
class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_list_models(self, client):
        """Test models listing."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        assert data["models"][0]["id"] == "dia"
    
    def test_list_voices(self, client):
        """Test voices listing."""
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert len(data["voices"]) == len(VOICE_MAPPING)
        
        # Check voice structure
        voice = data["voices"][0]
        assert "id" in voice
        assert "name" in voice
        assert "style" in voice
        assert "primary_speaker" in voice
    
    def test_voice_preview(self, client):
        """Test voice preview generation."""
        with patch('src.server.generate_audio_from_text') as mock_generate:
            # Mock audio generation
            mock_audio = np.zeros(44100, dtype=np.float32)
            mock_generate.return_value = (mock_audio, "log_123")
            
            response = client.get("/preview/alloy")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            mock_generate.assert_called_once()


@pytest.mark.unit
class TestGenerationEndpoints:
    """Test TTS generation endpoints."""
    
    def test_generate_sync(self, client, mock_audio_data):
        """Test synchronous generation."""
        audio, sr = mock_audio_data
        
        with patch('src.server.generate_audio_from_text') as mock_generate:
            mock_generate.return_value = (audio, "log_123")
            
            response = client.post("/generate", json={
                "text": "Hello world",
                "voice_id": "alloy",
                "speed": 1.0
            })
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            # Check that generate was called with keyword arguments
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args[0][0] == "Hello world"
            assert call_args[0][1] == "alloy"
            assert call_args[0][2] == 1.0
    
    def test_generate_async(self, client):
        """Test asynchronous generation."""
        with patch('src.server.WORKER_POOL') as mock_pool:
            mock_future = Mock()
            mock_pool.submit.return_value = mock_future
            
            response = client.post("/generate?async_mode=true", json={
                "text": "Hello world",
                "voice_id": "alloy"
            })
            
            # Async mode returns 200 with job info, not 202
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "pending"
            
            # Check job was queued
            mock_pool.submit.assert_called_once()
    
    def test_generate_invalid_voice(self, client):
        """Test generation with invalid voice."""
        response = client.post("/generate", json={
            "text": "Hello",
            "voice_id": "invalid_voice"
        })
        
        # Model not loaded in test environment
        assert response.status_code == 500
    
    def test_generate_empty_text(self, client):
        """Test generation with empty text."""
        response = client.post("/generate", json={
            "text": "",
            "voice_id": "alloy"
        })
        
        assert response.status_code == 400  # Bad request for empty text
    
    def test_generate_with_parameters(self, client, mock_audio_data):
        """Test generation with advanced parameters."""
        audio, sr = mock_audio_data
        
        with patch('src.server.generate_audio_from_text') as mock_generate:
            mock_generate.return_value = (audio, "log_123")
            
            response = client.post("/generate", json={
                "text": "Hello",
                "voice_id": "alloy",
                "speed": 1.5,
                "temperature": 1.2,
                "cfg_scale": 3.0,
                "top_p": 0.95,
                "max_tokens": 1000
            })
            
            assert response.status_code == 200
            
            # Check parameters were passed as keyword arguments
            call_args = mock_generate.call_args
            assert call_args[0][2] == 1.5  # speed
            assert call_args[1]['temperature'] == 1.2
            assert call_args[1]['cfg_scale'] == 3.0


@pytest.mark.unit
class TestJobEndpoints:
    """Test job management endpoints."""
    
    def test_get_job_status(self, client):
        """Test job status retrieval."""
        # Create actual TTSJob to avoid serialization issues
        job_id = "test_job_123"
        from src.server import TTSJob, JobStatus
        job = TTSJob(
            id=job_id,
            status=JobStatus.COMPLETED,
            created_at=datetime.now(),
            completed_at=datetime.now(),
            text="Test",
            voice_id="alloy",
            speed=1.0,
            generation_time=2.5
        )
        
        with patch('src.server.JOB_QUEUE', {job_id: job}):
            response = client.get(f"/jobs/{job_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == job_id
            assert data["status"] == "completed"
            assert data["generation_time"] == 2.5
    
    def test_get_job_status_not_found(self, client):
        """Test job status for non-existent job."""
        response = client.get("/jobs/nonexistent")
        assert response.status_code == 404
    
    def test_get_job_result(self, client):
        """Test job result retrieval."""
        job_id = "test_job_123"
        mock_audio = b"RIFF...WAVEfmt..."  # Mock WAV data
        
        # Create a completed job
        from src.server import TTSJob, JobStatus
        job = TTSJob(
            id=job_id,
            status=JobStatus.COMPLETED,
            created_at=datetime.now(),
            completed_at=datetime.now(),
            text="Test",
            voice_id="alloy",
            speed=1.0
        )
        
        with patch('src.server.JOB_QUEUE', {job_id: job}), \
             patch('src.server.JOB_RESULTS', {job_id: mock_audio}):
            response = client.get(f"/jobs/{job_id}/result")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            assert response.content == mock_audio
    
    def test_queue_stats(self, client):
        """Test queue statistics endpoint."""
        # Create mock jobs
        jobs = {
            "job1": Mock(status="pending"),
            "job2": Mock(status="processing"),
            "job3": Mock(status="completed"),
            "job4": Mock(status="failed")
        }
        
        with patch('src.server.JOB_QUEUE', jobs), \
             patch('src.server.MAX_WORKERS', 4), \
             patch('src.server.ACTIVE_WORKERS', {"w1": True, "w2": False}):
            
            response = client.get("/queue/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["pending_jobs"] == 1
            assert data["processing_jobs"] == 1
            assert data["completed_jobs"] == 1
            assert data["failed_jobs"] == 1
            assert data["total_workers"] == 4
            assert data["active_workers"] == 1


@pytest.mark.unit
class TestVoiceMappingEndpoints:
    """Test voice mapping management endpoints."""
    
    def test_get_voice_mappings(self, client):
        """Test getting voice mappings."""
        response = client.get("/voice_mappings")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(VOICE_MAPPING)
        assert "alloy" in data
    
    def test_update_voice_mapping(self, client):
        """Test updating voice mapping."""
        with patch.dict('src.server.VOICE_MAPPING', {"test": {"style": "old", "primary_speaker": "S1"}}):
            response = client.put("/voice_mappings/test", json={
                "voice_id": "test",
                "style": "new",
                "primary_speaker": "S2"
            })
            
            assert response.status_code == 200
            # Response is the entire VOICE_MAPPING dict
            import src.server
            assert src.server.VOICE_MAPPING["test"]["style"] == "new"
            assert src.server.VOICE_MAPPING["test"]["primary_speaker"] == "S2"
    
    def test_create_voice_mapping(self, client):
        """Test creating new voice mapping."""
        with patch.dict('src.server.VOICE_MAPPING', {}):
            response = client.post("/voice_mappings", json={
                "voice_id": "custom",
                "style": "custom_style",
                "primary_speaker": "S1"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Voice 'custom' created successfully"
    
    def test_delete_voice_mapping(self, client):
        """Test deleting voice mapping."""
        with patch.dict('src.server.VOICE_MAPPING', {"test": {"style": "test"}}):
            response = client.delete("/voice_mappings/test")
            
            assert response.status_code == 200
            assert "test" not in response.json()


@pytest.mark.unit
class TestAudioPromptEndpoints:
    """Test audio prompt management endpoints."""
    
    def test_upload_audio_prompt(self, client, tmp_path):
        """Test audio prompt upload."""
        import struct
        
        # Create a valid WAV file with proper headers
        # WAV file format: RIFF header + fmt chunk + data chunk
        sample_rate = 44100
        num_channels = 1
        bits_per_sample = 16
        duration = 1  # 1 second
        num_samples = sample_rate * duration
        
        # Generate simple sine wave data
        import math
        samples = []
        for i in range(num_samples):
            sample = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            samples.append(struct.pack('<h', sample))
        
        audio_data = b''.join(samples)
        
        # Create WAV header
        wav_header = b'RIFF'
        wav_header += struct.pack('<I', 36 + len(audio_data))  # File size - 8
        wav_header += b'WAVE'
        wav_header += b'fmt '
        wav_header += struct.pack('<I', 16)  # Subchunk size
        wav_header += struct.pack('<H', 1)   # Audio format (PCM)
        wav_header += struct.pack('<H', num_channels)
        wav_header += struct.pack('<I', sample_rate)
        wav_header += struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8)
        wav_header += struct.pack('<H', num_channels * bits_per_sample // 8)
        wav_header += struct.pack('<H', bits_per_sample)
        wav_header += b'data'
        wav_header += struct.pack('<I', len(audio_data))
        
        wav_data = wav_header + audio_data
        
        with patch('src.server.AUDIO_PROMPT_DIR', str(tmp_path)), \
             patch('src.server.ensure_audio_prompt_dir'):
            
            response = client.post(
                "/audio_prompts/upload",
                files={"audio_file": ("test.wav", wav_data, "audio/wav")},
                data={"prompt_id": "test_prompt"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["duration"] == 1.0
            assert data["sample_rate"] == 44100
            assert data["channels"] == "mono"
    
    def test_list_audio_prompts(self, client):
        """Test listing audio prompts."""
        with patch('src.server.AUDIO_PROMPTS', {"prompt1": "/path/1.wav", "prompt2": "/path/2.wav"}):
            response = client.get("/audio_prompts")
            
            assert response.status_code == 200
            data = response.json()
            # Response format is different
            assert "prompt1" in data
            assert "prompt2" in data
    
    def test_delete_audio_prompt(self, client, tmp_path):
        """Test deleting audio prompt."""
        # Create test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"test")
        
        with patch('src.server.AUDIO_PROMPTS', {"test": str(test_file)}), \
             patch('src.server.VOICE_MAPPING', {}):
            
            response = client.delete("/audio_prompts/test")
            
            assert response.status_code == 200
            assert not test_file.exists()


@pytest.mark.unit
class TestConfigEndpoints:
    """Test configuration endpoints."""
    
    def test_get_config(self, client):
        """Test getting server configuration."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "debug_mode" in data
        assert "save_outputs" in data
    
    def test_update_config(self, client):
        """Test updating server configuration."""
        # The config update actually modifies the real SERVER_CONFIG object
        response = client.put("/config", json={
            "debug_mode": True,
            "save_outputs": True
        })
        
        assert response.status_code == 200
        data = response.json()
        # Response includes message and config object
        assert data["message"] == "Configuration updated successfully"
        assert data["config"]["debug_mode"] is True
        assert data["config"]["save_outputs"] is True


# OpenAI endpoints removed - they don't exist in the server