"""Unit tests for audio processing functions."""

import pytest
import numpy as np
import soundfile as sf
import io
import os
import unittest.mock
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime, timedelta

from src.server import (
    preprocess_text, generate_audio_from_text, process_tts_job,
    ensure_output_dir, ensure_audio_prompt_dir, cleanup_old_files,
    VOICE_MAPPING
)


@pytest.mark.unit
class TestTextPreprocessing:
    """Test text preprocessing functions."""
    
    def test_preprocess_text_no_tags_with_voice(self):
        """Test preprocessing text without tags using voice mapping."""
        # Test alloy voice (S1)
        result = preprocess_text("Hello world", "alloy")
        assert result == "[S1] Hello world [S1]"
        
        # Test fable voice (S2)
        result = preprocess_text("Hello world", "fable")
        assert result == "[S2] Hello world [S2]"
    
    def test_preprocess_text_no_tags_with_role(self):
        """Test preprocessing text with role override."""
        # User role should use S1
        result = preprocess_text("Hello world", "fable", role="user")
        assert result == "[S1] Hello world [S1]"
        
        # Assistant role should use S2
        result = preprocess_text("Hello world", "alloy", role="assistant")
        assert result == "[S2] Hello world [S2]"
        
        # System role should also use S2 (same as assistant)
        result = preprocess_text("Hello world", "fable", role="system")
        assert result == "[S2] Hello world [S2]"
    
    def test_preprocess_text_with_existing_tags(self):
        """Test preprocessing text that already has speaker tags."""
        # Text with S1 tag should get closing tag
        result = preprocess_text("[S1] Hello world", "alloy")
        assert result == "[S1] Hello world [S1]"
        
        # Text with S2 tag should get closing tag
        result = preprocess_text("[S2] Hello world", "fable")
        assert result == "[S2] Hello world [S2]"
        
        # Text with complete tags should not be modified
        result = preprocess_text("[S1] Hello world [S1]", "alloy")
        assert result == "[S1] Hello world [S1]"
    
    def test_preprocess_text_unknown_voice(self):
        """Test preprocessing with unknown voice ID."""
        # Should default to alloy (S1)
        result = preprocess_text("Hello world", "unknown_voice")
        assert result == "[S1] Hello world [S1]"
    
    def test_preprocess_text_mixed_tags(self):
        """Test preprocessing text with mixed speaker tags."""
        # Should not add tags if already present
        text = "[S1] Hello [S2] World [S1]"
        result = preprocess_text(text, "alloy")
        assert result == text  # No modification when tags exist


@pytest.mark.unit
class TestAudioGeneration:
    """Test audio generation functions."""
    
    @patch('src.server.get_model_for_worker')
    @patch('src.server.preprocess_text')
    def test_generate_audio_from_text_basic(self, mock_preprocess, mock_get_model, mock_dia_model):
        """Test basic audio generation."""
        # Setup mocks
        mock_preprocess.return_value = "[S1] Test text [S1]"
        mock_get_model.return_value = (mock_dia_model, 0)
        
        # Mock audio output
        expected_audio = np.zeros(44100, dtype=np.float32)
        mock_dia_model.generate.return_value = expected_audio
        
        # Test generation
        audio, log_id = generate_audio_from_text("Test text", "alloy", speed=1.0)
        
        # Verify
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100
        assert isinstance(log_id, str)
        mock_preprocess.assert_called_once_with("Test text", "alloy", None)
        mock_dia_model.generate.assert_called_once()
    
    @patch('src.server.get_model_for_worker')
    def test_generate_audio_with_speed_adjustment(self, mock_get_model, mock_dia_model):
        """Test audio generation with speed adjustment."""
        mock_get_model.return_value = (mock_dia_model, 0)
        
        # Mock audio output (1 second at 44.1kHz)
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        mock_dia_model.generate.return_value = original_audio
        
        # Test with 2x speed
        audio, _ = generate_audio_from_text("Test", "alloy", speed=2.0)
        
        # Audio should be half the length
        assert len(audio) == 22050
        
        # Test with 0.5x speed
        audio, _ = generate_audio_from_text("Test", "alloy", speed=0.5)
        
        # Audio should be double the length
        assert len(audio) == 88200
    
    @patch('src.server.get_model_for_worker')
    @patch('src.server.AUDIO_PROMPTS', {"test_prompt.wav": "/tmp/test_prompt.wav"})
    @patch('os.path.exists', return_value=True)
    def test_generate_audio_with_audio_prompt(self, mock_exists, mock_get_model, mock_dia_model):
        """Test audio generation with audio prompt."""
        mock_get_model.return_value = (mock_dia_model, 0)
        mock_dia_model.generate.return_value = np.zeros(44100)
        
        # Update voice mapping to include audio prompt
        with patch.dict('src.server.VOICE_MAPPING', {
            "custom": {
                "style": "custom",
                "primary_speaker": "S1",
                "audio_prompt": "test_prompt.wav",
                "audio_prompt_transcript": "This is a test"
            }
        }):
            audio, _ = generate_audio_from_text("Hello", "custom")
            
            # Check that audio prompt was passed
            call_args = mock_dia_model.generate.call_args
            assert call_args[1]["audio_prompt"] == "/tmp/test_prompt.wav"
            
            # Check that transcript was prepended
            assert "This is a test" in call_args[0][0]
    
    @patch('src.server.get_model_for_worker')
    @patch('src.server.SERVER_CONFIG')
    @patch('src.server.ensure_output_dir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('soundfile.write')
    @patch('os.path.getsize', return_value=1024)  # Mock file size
    def test_generate_audio_save_output(self, mock_getsize, mock_sf_write, mock_makedirs, mock_exists,
                                       mock_ensure_dir, mock_config, mock_get_model, mock_dia_model):
        """Test saving audio output to file."""
        # Configure to save outputs
        mock_config.save_outputs = True
        mock_config.debug_mode = False
        mock_config.show_prompts = False
        mock_get_model.return_value = (mock_dia_model, 0)
        mock_dia_model.generate.return_value = np.zeros(44100)
        
        # Mock the file path generation to avoid Windows path issues
        with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            audio, log_id = generate_audio_from_text("Test", "alloy", speed=1.0)
        
        # Check that output was saved
        mock_ensure_dir.assert_called_once()
        mock_sf_write.assert_called()
        
        # Verify file format
        call_args = mock_sf_write.call_args
        assert call_args[1]["format"] == "WAV"
        assert call_args[1]["subtype"] == "PCM_16"
        assert call_args[0][1].shape == (44100,)  # Audio data
        assert call_args[0][2] == 44100  # Sample rate


@pytest.mark.unit
class TestJobProcessing:
    """Test job processing functions."""
    
    @patch('src.server.JOB_QUEUE')
    @patch('src.server.JOB_RESULTS')
    @patch('src.server.generate_audio_from_text')
    @patch('src.server.ACTIVE_WORKERS', {})
    def test_process_tts_job_success(self, mock_generate, mock_results, mock_queue):
        """Test successful job processing."""
        # Setup mock job
        job = Mock()
        job.text = "Test text"
        job.voice_id = "alloy"
        job.speed = 1.0
        job.temperature = None
        job.cfg_scale = None
        job.top_p = None
        job.max_tokens = None
        job.use_torch_compile = None
        job.role = None
        job.status = "pending"
        
        mock_queue.get.return_value = job
        
        # Mock audio generation
        audio = np.zeros(44100, dtype=np.float32)
        mock_generate.return_value = (audio, "log_123")
        
        # Process job
        process_tts_job("job_123")
        
        # Verify job was processed
        assert job.status == "completed"
        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.generation_time > 0
        
        # Verify audio was stored
        # Check that result was stored (mock_results is a MagicMock)
        mock_results.__setitem__.assert_called_once_with("job_123", unittest.mock.ANY)
        # Verify it was called with bytes
        stored_data = mock_results.__setitem__.call_args[0][1]
        assert isinstance(stored_data, bytes)
    
    @patch('src.server.JOB_QUEUE')
    @patch('src.server.generate_audio_from_text')
    @patch('src.server.ACTIVE_WORKERS', {})
    def test_process_tts_job_failure(self, mock_generate, mock_queue):
        """Test job processing with failure."""
        # Setup mock job
        job = Mock()
        job.text = "Test text"
        job.voice_id = "alloy"
        job.speed = 1.0
        
        mock_queue.get.return_value = job
        
        # Mock generation failure
        mock_generate.side_effect = Exception("Generation failed")
        
        # Process job
        process_tts_job("job_123")
        
        # Verify job failed
        assert job.status == "failed"
        assert job.error_message == "Generation failed"
        assert job.completed_at is not None


@pytest.mark.unit
class TestDirectoryManagement:
    """Test directory and file management functions."""
    
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('src.server.SERVER_CONFIG')
    def test_ensure_output_dir_creates(self, mock_config, mock_makedirs, mock_exists):
        """Test output directory creation."""
        mock_config.save_outputs = True
        
        ensure_output_dir()
        
        mock_makedirs.assert_called_once_with("audio_outputs", exist_ok=True)
    
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('src.server.SERVER_CONFIG')
    def test_ensure_output_dir_exists(self, mock_config, mock_makedirs, mock_exists):
        """Test when output directory already exists."""
        mock_config.save_outputs = True
        
        ensure_output_dir()
        
        mock_makedirs.assert_not_called()
    
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    def test_ensure_audio_prompt_dir(self, mock_makedirs, mock_exists):
        """Test audio prompt directory creation."""
        ensure_audio_prompt_dir()
        
        mock_makedirs.assert_called_once_with("audio_prompts", exist_ok=True)
    
    @patch('src.server.SERVER_CONFIG')
    @patch('os.path.exists', return_value=True)
    @patch('os.listdir')
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.getctime')
    @patch('os.remove')
    def test_cleanup_old_files(self, mock_remove, mock_getctime, mock_isfile,
                               mock_listdir, mock_exists, mock_config):
        """Test cleanup of old audio files."""
        mock_config.save_outputs = True
        mock_config.output_retention_hours = 1
        
        # Setup mock files
        mock_listdir.return_value = ["old_file.wav", "new_file.wav"]
        
        # Old file (2 hours ago)
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        # New file (30 minutes ago)
        new_time = (datetime.now() - timedelta(minutes=30)).timestamp()
        
        mock_getctime.side_effect = [old_time, new_time]
        
        cleanup_old_files()
        
        # Only old file should be removed
        mock_remove.assert_called_once_with(os.path.join("audio_outputs", "old_file.wav"))