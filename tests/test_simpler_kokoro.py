"""
Comprehensive pytest test suite for SimplerKokoro library.

Run with: pytest test_simpler_kokoro.py -v
Run with coverage: pytest test_simpler_kokoro.py -v --cov=simpler_kokoro --cov-report=html
"""

import pytest
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

# Import the module to test
from simpler_kokoro import (
    SimplerKokoro,
    Voice,
    GenerationConfig,
    KokoroException,
    VoiceNotFoundError,
    ModelDownloadError,
    setup_logger,
    main
)


# Fixtures
@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_voice():
    """Create a mock Voice object."""
    return Voice(
        name="af_alloy",
        display_name="Alloy",
        gender="Female",
        lang_code="a",
        model_path="/path/to/af_alloy.pt"
    )


@pytest.fixture
def mock_voices():
    """Create a list of mock Voice objects."""
    return [
        Voice(
            name="af_alloy",
            display_name="Alloy",
            gender="Female",
            lang_code="a",
            model_path="/path/to/af_alloy.pt"
        ),
        Voice(
            name="am_adam",
            display_name="Adam",
            gender="Male",
            lang_code="a",
            model_path="/path/to/am_adam.pt"
        ),
        Voice(
            name="bf_bella",
            display_name="Bella",
            gender="Female",
            lang_code="b",
            model_path="/path/to/bf_bella.pt"
        )
    ]


@pytest.fixture
def mock_kokoro_module():
    """Create a mock kokoro module."""
    mock_module = MagicMock()
    mock_pipeline = MagicMock()
    mock_module.KPipeline = Mock(return_value=mock_pipeline)
    return mock_module


@pytest.fixture
def mock_audio_data():
    """Create mock audio data."""
    mock_data = MagicMock()
    mock_data.audio = np.random.randn(24000)  # 1 second of audio
    mock_data.graphemes = "Test text"
    
    # Mock tokens
    mock_token1 = MagicMock()
    mock_token1.text = "Test"
    mock_token1.start_ts = 0.0
    mock_token1.end_ts = 0.5
    
    mock_token2 = MagicMock()
    mock_token2.text = "text"
    mock_token2.start_ts = 0.5
    mock_token2.end_ts = 1.0
    
    mock_data.tokens = [mock_token1, mock_token2]
    return mock_data


# Tests for Voice dataclass
class TestVoice:
    """Tests for Voice dataclass."""
    
    def test_voice_creation(self):
        """Test creating a Voice object."""
        voice = Voice(
            name="af_alloy",
            display_name="Alloy",
            gender="Female",
            lang_code="a",
            model_path="/path/to/model.pt"
        )
        
        assert voice.name == "af_alloy"
        assert voice.display_name == "Alloy"
        assert voice.gender == "Female"
        assert voice.lang_code == "a"
        assert voice.model_path == "/path/to/model.pt"
    
    def test_voice_str(self):
        """Test Voice __str__ method."""
        voice = Voice(
            name="af_alloy",
            display_name="Alloy",
            gender="Female",
            lang_code="a",
            model_path="/path/to/model.pt"
        )
        
        assert str(voice) == "Alloy (Female, a)"


# Tests for GenerationConfig dataclass
class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig(
            text="Test text",
            voice="af_alloy",
            output_path="output.wav"
        )
        
        assert config.text == "Test text"
        assert config.voice == "af_alloy"
        assert config.output_path == "output.wav"
        assert config.speed == 1.0
        assert config.write_subtitles is False
        assert config.subtitles_path == "subtitles.srt"
        assert config.subtitles_word_level is False
        assert config.split_pattern == r'\.\s+|\n'
        assert config.sample_rate == 24000
    
    def test_config_custom_values(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            text="Test",
            voice="af_alloy",
            output_path="out.wav",
            speed=1.5,
            write_subtitles=True,
            subtitles_path="subs.srt",
            subtitles_word_level=True,
            split_pattern=r'\n',
            sample_rate=48000
        )
        
        assert config.speed == 1.5
        assert config.write_subtitles is True
        assert config.subtitles_path == "subs.srt"
        assert config.subtitles_word_level is True
        assert config.split_pattern == r'\n'
        assert config.sample_rate == 48000


# Tests for setup_logger
class TestSetupLogger:
    """Tests for setup_logger function."""
    
    def test_logger_creation(self):
        """Test logger is created correctly."""
        logger = setup_logger(level=logging.DEBUG, name="test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0
    
    def test_logger_no_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't add duplicate handlers."""
        logger = setup_logger(name="test_logger2")
        handler_count = len(logger.handlers)
        
        logger = setup_logger(name="test_logger2")
        assert len(logger.handlers) == handler_count


# Tests for SimplerKokoro initialization
class TestSimplerKokoroInit:
    """Tests for SimplerKokoro initialization."""
    
    def test_init_skip_download(self, temp_models_dir):
        """Test initialization with skip_download=True."""
        sk = SimplerKokoro(
            models_dir=temp_models_dir,
            skip_download=True
        )
        
        assert sk.device == "cpu"
        assert str(sk.models_dir) == temp_models_dir
        assert sk.repo == SimplerKokoro.DEFAULT_REPO
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_init_creates_directories(self, mock_list_files, temp_models_dir):
        """Test that initialization creates necessary directories."""
        mock_list_files.return_value = []
        
        with patch('simpler_kokoro.SimplerKokoro.download_models'):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(KokoroException, match="Kokoro module not found"):
                    SimplerKokoro(models_dir=temp_models_dir, auto_download=False)
        
        assert Path(temp_models_dir, 'kokoro').exists()
        assert Path(temp_models_dir, 'voices').exists()
    
    def test_init_custom_parameters(self, temp_models_dir):
        """Test initialization with custom parameters."""
        sk = SimplerKokoro(
            device="cuda",
            models_dir=temp_models_dir,
            repo="custom/repo",
            skip_download=True
        )
        
        assert sk.device == "cuda"
        assert sk.repo == "custom/repo"


# Tests for directory management
class TestDirectoryManagement:
    """Tests for directory management methods."""
    
    def test_ensure_models_dirs(self, temp_models_dir):
        """Test that ensure_models_dirs creates required directories."""
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        sk.ensure_models_dirs()
        
        assert sk.kokoro_model_dir.exists()
        assert sk.kokoro_voices_path.exists()


# Tests for voice listing
class TestVoiceListing:
    """Tests for voice listing functionality."""
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_list_voices_remote(self, mock_list_files):
        """Test listing voices from remote repository."""
        mock_list_files.return_value = [
            "voices/af_alloy.pt",
            "voices/am_adam.pt",
            "voices/bf_bella.pt",
            "other_file.txt"
        ]
        
        voices = SimplerKokoro.list_voices_remote("test/repo")
        
        assert len(voices) == 3
        assert all(isinstance(v, Voice) for v in voices)
        assert voices[0].name == "af_alloy"
        assert voices[0].gender == "Female"
        assert voices[1].name == "am_adam"
        assert voices[1].gender == "Male"
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_list_voices_remote_error(self, mock_list_files):
        """Test list_voices_remote handles errors."""
        mock_list_files.side_effect = Exception("Network error")
        
        with pytest.raises(ModelDownloadError):
            SimplerKokoro.list_voices_remote("test/repo")
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_list_voices_remote_invalid_files(self, mock_list_files):
        """Test list_voices_remote skips invalid voice files."""
        mock_list_files.return_value = [
            "voices/af_alloy.pt",
            "voices/xx.pt",  # Too short
            "voices/am_adam.pt"
        ]
        
        voices = SimplerKokoro.list_voices_remote()
        
        assert len(voices) == 2
        assert all(v.name in ["af_alloy", "am_adam"] for v in voices)
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_list_voices_sorting(self, mock_list_files):
        """Test that voices are sorted correctly."""
        mock_list_files.return_value = [
            "voices/bf_bella.pt",
            "voices/af_alloy.pt",
            "voices/am_adam.pt"
        ]
        
        voices = SimplerKokoro.list_voices_remote()
        
        # Should be sorted by lang_code, gender, display_name
        assert voices[0].name == "af_alloy"
        assert voices[1].name == "am_adam"
        assert voices[2].name == "bf_bella"


# Tests for model downloading
class TestModelDownloading:
    """Tests for model downloading functionality."""
    
    @patch('simpler_kokoro.hf.hf_hub_download')
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_download_models_main(self, mock_list_files, mock_download, temp_models_dir):
        """Test downloading main model."""
        mock_list_files.return_value = ["voices/af_alloy.pt"]
        
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        sk.download_models()
        
        # Check that main model download was called
        calls = [call for call in mock_download.call_args_list 
                if call[1].get('filename') == 'kokoro-v1_0.pth']
        assert len(calls) > 0
    
    @patch('simpler_kokoro.hf.hf_hub_download')
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_download_models_voices(self, mock_list_files, mock_download, temp_models_dir):
        """Test downloading voice models."""
        mock_list_files.return_value = [
            "voices/af_alloy.pt",
            "voices/am_adam.pt"
        ]
        
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        sk.download_models()
        
        # Check that voice downloads were called
        voice_calls = [call for call in mock_download.call_args_list 
                      if 'voices/' in str(call[1].get('filename', ''))]
        assert len(voice_calls) >= 2
    
    @patch('simpler_kokoro.hf.hf_hub_download')
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_download_models_error_handling(self, mock_list_files, mock_download, temp_models_dir):
        """Test error handling during model download."""
        mock_list_files.return_value = ["voices/af_alloy.pt"]
        mock_download.side_effect = Exception("Download failed")
        
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        
        with pytest.raises(ModelDownloadError):
            sk.download_models()


# Tests for voice retrieval
class TestVoiceRetrieval:
    """Tests for get_voice method."""
    
    def test_get_voice_success(self, mock_voices):
        """Test successful voice retrieval."""
        sk = SimplerKokoro(skip_download=True)
        sk.voices = mock_voices
        
        voice = sk.get_voice("af_alloy")
        
        assert voice.name == "af_alloy"
        assert voice.display_name == "Alloy"
    
    def test_get_voice_not_found(self, mock_voices):
        """Test VoiceNotFoundError is raised for missing voice."""
        sk = SimplerKokoro(skip_download=True)
        sk.voices = mock_voices
        
        with pytest.raises(VoiceNotFoundError, match="Voice 'nonexistent' not found"):
            sk.get_voice("nonexistent")


# Tests for generation
class TestGeneration:
    """Tests for speech generation."""
    
    @patch('simpler_kokoro.SimplerKokoro._generate_with_config')
    def test_generate_creates_config(self, mock_generate, mock_voices):
        """Test that generate method creates proper config."""
        sk = SimplerKokoro(skip_download=True)
        sk.voices = mock_voices
        
        sk.generate(
            text="Test text",
            voice="af_alloy",
            output_path="output.wav",
            speed=1.5,
            write_subtitles=True
        )
        
        mock_generate.assert_called_once()
        config = mock_generate.call_args[0][0]
        
        assert isinstance(config, GenerationConfig)
        assert config.text == "Test text"
        assert config.voice == "af_alloy"
        assert config.speed == 1.5
        assert config.write_subtitles is True
    
    def test_generate_without_kokoro_module(self, mock_voices):
        """Test generation fails without kokoro module."""
        sk = SimplerKokoro(skip_download=True)
        sk.voices = mock_voices
        sk.kokoro = None
        
        with pytest.raises(KokoroException, match="Kokoro module not initialized"):
            sk.generate(
                text="Test",
                voice="af_alloy",
                output_path="output.wav"
            )
    
    def test_generate_with_invalid_voice(self):
        """Test generation fails with invalid voice."""
        sk = SimplerKokoro(skip_download=True)
        sk.voices = []
        sk.kokoro = MagicMock()
        
        with pytest.raises(VoiceNotFoundError):
            sk.generate(
                text="Test",
                voice="nonexistent",
                output_path="output.wav"
            )


# Tests for audio processing
class TestAudioProcessing:
    """Tests for audio processing methods."""
    
    @patch('simpler_kokoro.sf.write')
    @patch('simpler_kokoro.sf.read')
    def test_process_audio_chunks(self, mock_read, mock_write, temp_models_dir, mock_audio_data):
        """Test audio chunk processing."""
        mock_read.return_value = (np.random.randn(24000), 24000)
        
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        
        config = GenerationConfig(
            text="Test",
            voice="af_alloy",
            output_path=os.path.join(temp_models_dir, "output.wav")
        )
        
        generator = [mock_audio_data]
        
        sk._process_audio_chunks(generator, config)
        
        # Check that write was called
        assert mock_write.called
    
    @patch('simpler_kokoro.sf.write')
    @patch('simpler_kokoro.sf.read')
    def test_process_audio_chunks_with_subtitles(self, mock_read, mock_write, 
                                                  temp_models_dir, mock_audio_data):
        """Test audio processing with subtitle generation."""
        mock_read.return_value = (np.random.randn(24000), 24000)
        
        sk = SimplerKokoro(models_dir=temp_models_dir, skip_download=True)
        
        config = GenerationConfig(
            text="Test",
            voice="af_alloy",
            output_path=os.path.join(temp_models_dir, "output.wav"),
            write_subtitles=True,
            subtitles_path=os.path.join(temp_models_dir, "subs.srt")
        )
        
        generator = [mock_audio_data]
        
        sk._process_audio_chunks(generator, config)
        
        # Check that subtitle file was created
        assert Path(config.subtitles_path).exists()


# Tests for subtitle writing
class TestSubtitleWriting:
    """Tests for subtitle writing functionality."""
    
    def test_write_srt_subtitles(self, temp_models_dir):
        """Test SRT subtitle writing."""
        subtitles = {
            0: {'text': 'First line', 'start': 0.0, 'end': 1.5},
            1: {'text': 'Second line', 'start': 1.5, 'end': 3.0}
        }
        
        output_path = os.path.join(temp_models_dir, "test.srt")
        SimplerKokoro._write_srt_subtitles(subtitles, output_path)
        
        assert Path(output_path).exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "First line" in content
            assert "Second line" in content
            assert "00:00:00,000 --> 00:00:01,500" in content
    
    def test_write_srt_subtitles_creates_dir(self, temp_models_dir):
        """Test that subtitle writing creates parent directory."""
        output_path = os.path.join(temp_models_dir, "subdir", "test.srt")
        subtitles = {0: {'text': 'Test', 'start': 0.0, 'end': 1.0}}
        
        SimplerKokoro._write_srt_subtitles(subtitles, output_path)
        
        assert Path(output_path).exists()


# Tests for CLI
class TestCLI:
    """Tests for command-line interface."""
    
    @patch('simpler_kokoro.SimplerKokoro.list_voices_remote')
    @patch('sys.argv', ['simpler_kokoro', 'list-voices', '--remote'])
    def test_cli_list_voices_remote(self, mock_list_voices, capsys):
        """Test CLI list-voices command with --remote flag."""
        mock_list_voices.return_value = [
            Voice("af_alloy", "Alloy", "Female", "a", "/path/to/model.pt")
        ]
        
        main()
        
        captured = capsys.readouterr()
        assert "af_alloy" in captured.out
        assert "Alloy" in captured.out
    
    @patch('simpler_kokoro.SimplerKokoro')
    @patch('sys.argv', ['simpler_kokoro', 'list-voices', '--format', 'json'])
    def test_cli_list_voices_json(self, mock_sk_class, capsys):
        """Test CLI list-voices with JSON format."""
        mock_sk = MagicMock()
        mock_sk.voices = [
            Voice("af_alloy", "Alloy", "Female", "a", "/path/to/model.pt")
        ]
        mock_sk_class.return_value = mock_sk
        
        main()
        
        captured = capsys.readouterr()
        assert '"name": "af_alloy"' in captured.out
    
    @patch('simpler_kokoro.SimplerKokoro')
    @patch('sys.argv', ['simpler_kokoro', 'list-voices', '--format', 'names'])
    def test_cli_list_voices_names(self, mock_sk_class, capsys):
        """Test CLI list-voices with names format."""
        mock_sk = MagicMock()
        mock_sk.voices = [
            Voice("af_alloy", "Alloy", "Female", "a", "/path/to/model.pt"),
            Voice("am_adam", "Adam", "Male", "a", "/path/to/model2.pt")
        ]
        mock_sk_class.return_value = mock_sk
        
        main()
        
        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        assert "af_alloy" in lines
        assert "am_adam" in lines
    
    @patch('simpler_kokoro.SimplerKokoro')
    @patch('sys.argv', [
        'simpler_kokoro', 'generate',
        '--text', 'Test text',
        '--voice', 'af_alloy',
        '--output', 'output.wav'
    ])
    def test_cli_generate(self, mock_sk_class, capsys):
        """Test CLI generate command."""
        mock_sk = MagicMock()
        mock_sk_class.return_value = mock_sk
        
        main()
        
        mock_sk.generate.assert_called_once()
        captured = capsys.readouterr()
        assert "Audio saved to output.wav" in captured.out
    
    @patch('simpler_kokoro.SimplerKokoro')
    @patch('sys.argv', [
        'simpler_kokoro', 'generate',
        '--text', 'Test',
        '--voice', 'af_alloy',
        '--output', 'out.wav',
        '--write_subtitles',
        '--subtitles_path', 'subs.srt'
    ])
    def test_cli_generate_with_subtitles(self, mock_sk_class, capsys):
        """Test CLI generate with subtitles."""
        mock_sk = MagicMock()
        mock_sk_class.return_value = mock_sk
        
        main()
        
        captured = capsys.readouterr()
        assert "Subtitles saved to subs.srt" in captured.out


# Tests for exception handling
class TestExceptionHandling:
    """Tests for custom exception classes."""
    
    def test_kokoro_exception(self):
        """Test KokoroException can be raised and caught."""
        with pytest.raises(KokoroException, match="Test error"):
            raise KokoroException("Test error")
    
    def test_voice_not_found_error(self):
        """Test VoiceNotFoundError inheritance."""
        with pytest.raises(KokoroException):
            raise VoiceNotFoundError("Voice not found")
    
    def test_model_download_error(self):
        """Test ModelDownloadError inheritance."""
        with pytest.raises(KokoroException):
            raise ModelDownloadError("Download failed")


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @patch('simpler_kokoro.hf.list_repo_files')
    @patch('simpler_kokoro.hf.hf_hub_download')
    def test_full_initialization_workflow(self, mock_download, mock_list_files, temp_models_dir):
        """Test complete initialization workflow."""
        mock_list_files.return_value = ["voices/af_alloy.pt"]
        
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(KokoroException):
                SimplerKokoro(models_dir=temp_models_dir)
        
        # Verify directories were created
        assert Path(temp_models_dir, 'kokoro').exists()
        assert Path(temp_models_dir, 'voices').exists()


# Parametrized tests
class TestParametrized:
    """Parametrized tests for various scenarios."""
    
    @pytest.mark.parametrize("voice_name,expected_gender", [
        ("af_alloy", "Female"),
        ("am_adam", "Male"),
        ("bf_bella", "Female"),
        ("bm_bob", "Male"),
    ])
    def test_voice_gender_parsing(self, voice_name, expected_gender):
        """Test that voice gender is parsed correctly."""
        voice = Voice(
            name=voice_name,
            display_name=voice_name[3:].capitalize(),
            gender=expected_gender,
            lang_code=voice_name[0],
            model_path="/path/to/model.pt"
        )
        
        assert voice.gender == expected_gender
    
    @pytest.mark.parametrize("speed", [0.5, 1.0, 1.5, 2.0])
    def test_generation_speeds(self, speed):
        """Test generation config with different speeds."""
        config = GenerationConfig(
            text="Test",
            voice="af_alloy",
            output_path="out.wav",
            speed=speed
        )
        
        assert config.speed == speed
    
    @pytest.mark.parametrize("log_level", [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL
    ])
    def test_different_log_levels(self, log_level):
        """Test logger with different log levels."""
        logger = setup_logger(level=log_level, name=f"test_logger_{log_level}")
        assert logger.level == log_level


# Performance/edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_text_generation(self):
        """Test handling of empty text."""
        config = GenerationConfig(
            text="",
            voice="af_alloy",
            output_path="output.wav"
        )
        
        assert config.text == ""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "Test sentence. " * 1000
        config = GenerationConfig(
            text=long_text,
            voice="af_alloy",
            output_path="output.wav"
        )
        
        assert len(config.text) > 10000
    
    def test_special_characters_in_path(self, temp_models_dir):
        """Test handling paths with special characters."""
        special_path = os.path.join(temp_models_dir, "test file (1).wav")
        config = GenerationConfig(
            text="Test",
            voice="af_alloy",
            output_path=special_path
        )
        
        assert config.output_path == special_path
    
    @patch('simpler_kokoro.hf.list_repo_files')
    def test_empty_voice_list(self, mock_list_files):
        """Test handling of empty voice list."""
        mock_list_files.return_value = []
        
        voices = SimplerKokoro.list_voices_remote()
        
        assert voices == []


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])