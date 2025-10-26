# Simpler Kokoro - A simplified interface for generating speech and subtitles using Kokoro voices

import os
import warnings
import tempfile
from typing import Optional, List, Dict, Generator
from pathlib import Path
from dataclasses import dataclass
import soundfile as sf
import huggingface_hub as hf
import logging


# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Voice:
    """Data class representing a Kokoro voice."""
    name: str
    display_name: str
    gender: str
    lang_code: str
    model_path: str
    
    def __str__(self) -> str:
        return f"{self.display_name} ({self.gender}, {self.lang_code})"


@dataclass
class GenerationConfig:
    """Configuration for speech generation."""
    text: str
    voice: str
    output_path: str
    speed: float = 1.0
    write_subtitles: bool = False
    subtitles_path: str = 'subtitles.srt'
    subtitles_word_level: bool = False
    split_pattern: str = r'\.\s+|\n'
    sample_rate: int = 24000


def setup_logger(level: int = logging.INFO, name: str = __name__) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


class KokoroException(Exception):
    """Base exception for SimplerKokoro errors."""
    pass


class VoiceNotFoundError(KokoroException):
    """Raised when a requested voice is not found."""
    pass


class ModelDownloadError(KokoroException):
    """Raised when model download fails."""
    pass


class SimplerKokoro:
    """
    SimplerKokoro provides a simplified interface for generating speech and subtitles using Kokoro voices.
    """
    
    DEFAULT_REPO = "hexgrad/Kokoro-82M"
    DEFAULT_MODELS_DIR = "models"
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        device: str = "cpu",
        models_dir: Optional[str] = None,
        repo: Optional[str] = None,
        log_level: int = logging.INFO,
        skip_download: bool = False,
        auto_download: bool = True
    ):
        """
        Initialize SimplerKokoro.
        
        Args:
            device: Device to use for inference (default: "cpu").
            models_dir: Directory to store model files (default: 'models').
            repo: HuggingFace repo to use for models (default: 'hexgrad/Kokoro-82M').
            log_level: Logging level (default: logging.INFO).
            skip_download: If True, do not download models or create directories (default: False).
            auto_download: If True, automatically download missing models (default: True).
        """
        global logger
        logger = setup_logger(log_level)
        
        self.device = device
        self.models_dir = Path(models_dir or self.DEFAULT_MODELS_DIR)
        self.repo = repo or self.DEFAULT_REPO
        self.auto_download = auto_download
        
        self.kokoro_model_dir = self.models_dir / 'kokoro'
        self.kokoro_model_path = self.kokoro_model_dir / 'kokoro-v1_0.pth'
        self.kokoro_voices_path = self.models_dir / 'voices'
        
        self.kokoro = None
        self.voices: List[Voice] = []
        
        if not skip_download:
            self._initialize()
    
    def _initialize(self):
        """Initialize the library by setting up directories and downloading models."""
        self.ensure_models_dirs()
        
        if self.auto_download:
            self.download_models()
        
        try:
            import kokoro
            self.kokoro = kokoro
            self.voices = self.list_voices()
            logger.info(f"Initialized with {len(self.voices)} voices available")
        except ImportError as e:
            logger.error(f"Failed to import kokoro module: {e}")
            raise KokoroException("Kokoro module not found. Please install it first.")
    
    def ensure_models_dirs(self):
        """Ensure the necessary model directories exist."""
        self.kokoro_model_dir.mkdir(parents=True, exist_ok=True)
        self.kokoro_voices_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Model directories ensured: {self.models_dir}")
    
    @staticmethod
    def list_voices_remote(repo: Optional[str] = None) -> List[Voice]:
        """
        Return a list of available Kokoro voices with metadata from HuggingFace only.
        
        Args:
            repo: HuggingFace repo to use for models.
            
        Returns:
            List of Voice objects.
        """
        repo = repo or SimplerKokoro.DEFAULT_REPO
        
        try:
            repo_files = hf.list_repo_files(repo)
            voice_files = [f for f in repo_files if f.startswith("voices/") and f.endswith(".pt")]
            voices = []
            
            for vf in voice_files:
                try:
                    voice_name = Path(vf).stem
                    
                    if len(voice_name) < 3:
                        logger.warning(f"Skipping invalid voice file: {vf}")
                        continue
                    
                    lang_code = voice_name[0]
                    gender = 'Male' if voice_name[1] == 'm' else 'Female'
                    display_name = voice_name[3:].capitalize()
                    
                    voices.append(Voice(
                        name=voice_name,
                        display_name=display_name,
                        gender=gender,
                        lang_code=lang_code,
                        model_path=vf
                    ))
                except Exception as e:
                    logger.error(f"Error parsing voice file {vf}: {e}")
            
            return sorted(voices, key=lambda v: (v.lang_code, v.gender, v.display_name))
            
        except Exception as e:
            logger.error(f"Error fetching voice list from HuggingFace Hub: {e}")
            raise ModelDownloadError(f"Failed to fetch voice list: {e}")
    
    def download_models(self, force: bool = False):
        """
        Download the Kokoro model files if they do not exist.
        
        Args:
            force: If True, re-download even if files exist.
        """
        # Download main model
        if force or not self.kokoro_model_path.exists():
            logger.info("Downloading main Kokoro model...")
            try:
                hf.hf_hub_download(
                    repo_id=self.repo,
                    filename="kokoro-v1_0.pth",
                    local_dir=str(self.kokoro_model_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("Main model downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading main Kokoro model: {e}")
                raise ModelDownloadError(f"Failed to download main model: {e}")
        
        # Download voice models
        try:
            repo_files = hf.list_repo_files(self.repo)
            voice_files = [f for f in repo_files if f.startswith('voices/') and f.endswith('.pt')]
            
            for voice_file in voice_files:
                local_path = self.models_dir / voice_file
                
                if force or not local_path.exists():
                    logger.info(f"Downloading voice model: {voice_file}")
                    try:
                        hf.hf_hub_download(
                            repo_id=self.repo,
                            filename=voice_file,
                            local_dir=str(self.models_dir),
                            local_dir_use_symlinks=False
                        )
                    except Exception as e:
                        logger.error(f"Error downloading voice model {voice_file}: {e}")
            
            logger.info("All voice models downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error fetching voice file list: {e}")
            raise ModelDownloadError(f"Failed to download voice models: {e}")
    
    def list_voices(self) -> List[Voice]:
        """
        Return a list of available Kokoro voices with metadata.
        
        Returns:
            List of Voice objects with local paths.
        """
        try:
            repo_files = hf.list_repo_files(self.repo)
            voice_files = [f for f in repo_files if f.startswith("voices/") and f.endswith(".pt")]
            voices = []
            
            for vf in voice_files:
                try:
                    voice_name = Path(vf).stem
                    
                    if len(voice_name) < 3:
                        continue
                    
                    lang_code = voice_name[0]
                    gender = 'Male' if voice_name[1] == 'm' else 'Female'
                    display_name = voice_name[3:].capitalize()
                    
                    voices.append(Voice(
                        name=voice_name,
                        display_name=display_name,
                        gender=gender,
                        lang_code=lang_code,
                        model_path=str(self.kokoro_voices_path / f"{voice_name}.pt")
                    ))
                except Exception as e:
                    logger.error(f"Error parsing voice file {vf}: {e}")
            
            return sorted(voices, key=lambda v: (v.lang_code, v.gender, v.display_name))
            
        except Exception as e:
            logger.error(f"Error fetching voice list: {e}")
            return []
    
    def get_voice(self, voice_name: str) -> Voice:
        """
        Get a Voice object by name.
        
        Args:
            voice_name: Name of the voice to retrieve.
            
        Returns:
            Voice object.
            
        Raises:
            VoiceNotFoundError: If the voice is not found.
        """
        for voice in self.voices:
            if voice.name == voice_name:
                return voice
        
        raise VoiceNotFoundError(f"Voice '{voice_name}' not found. Available voices: {[v.name for v in self.voices]}")
    
    def generate(
        self,
        text: str,
        voice: str,
        output_path: str,
        speed: float = 1.0,
        write_subtitles: bool = False,
        subtitles_path: str = 'subtitles.srt',
        subtitles_word_level: bool = False,
        split_pattern: str = r'\.\s+|\n'
    ):
        """
        Generate speech audio and optional subtitles from text using a Kokoro voice.

        Args:
            text: The input text to synthesize.
            voice: The Kokoro voice name (e.g., 'af_alloy').
            output_path: Path to save the combined output audio file.
            speed: Speech speed multiplier (default: 1.0).
            write_subtitles: Whether to write subtitles (default: False).
            subtitles_path: Path to save subtitles (default: 'subtitles.srt').
            subtitles_word_level: If True, subtitles are word-level; else, chunk-level.
            split_pattern: Regex pattern for splitting text into chunks.
            
        Raises:
            VoiceNotFoundError: If the voice is not found.
            KokoroException: If generation fails.
        """
        config = GenerationConfig(
            text=text,
            voice=voice,
            output_path=output_path,
            speed=speed,
            write_subtitles=write_subtitles,
            subtitles_path=subtitles_path,
            subtitles_word_level=subtitles_word_level,
            split_pattern=split_pattern
        )
        
        self._generate_with_config(config)
    
    def _generate_with_config(self, config: GenerationConfig):
        """Internal method to generate speech with a configuration object."""
        if not self.kokoro:
            raise KokoroException("Kokoro module not initialized")
        
        # Get voice information
        try:
            voice_obj = self.get_voice(config.voice)
        except VoiceNotFoundError:
            raise
        
        logger.info(f"Generating speech with voice: {voice_obj}")
        
        # Create Kokoro pipeline
        try:
            pipeline = self.kokoro.KPipeline(
                lang_code=voice_obj.lang_code,
                repo_id=self.repo
            )
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise KokoroException(f"Failed to create pipeline: {e}")
        
        # Load custom voice model
        voice_model = None
        if Path(voice_obj.model_path).exists():
            try:
                import torch
                voice_model = torch.load(voice_obj.model_path, weights_only=True)
                logger.debug(f"Loaded custom voice model from {voice_obj.model_path}")
            except Exception as e:
                logger.warning(f"Error loading custom model: {e}. Using default.")
        
        # Generate audio
        try:
            if voice_model is not None:
                generator = pipeline(
                    text=config.text,
                    voice=voice_model,
                    speed=config.speed,
                    split_pattern=config.split_pattern,
                )
            else:
                generator = pipeline(
                    text=config.text,
                    voice=config.voice,
                    speed=config.speed,
                    split_pattern=config.split_pattern,
                )
        except Exception as e:
            logger.error(f"Error creating generator: {e}")
            raise KokoroException(f"Failed to create generator: {e}")
        
        # Process audio chunks
        self._process_audio_chunks(generator, config)
        
        logger.info(f"Audio saved to {config.output_path}")
        if config.write_subtitles:
            logger.info(f"Subtitles saved to {config.subtitles_path}")
    
    def _process_audio_chunks(self, generator: Generator, config: GenerationConfig):
        """Process audio chunks and optionally generate subtitles."""
        import numpy as np
        
        subs = {}
        word_idx = 0
        audio_chunks = []
        cumulative_time = 0.0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, data in enumerate(generator):
                try:
                    chunk_duration = len(data.audio) / config.sample_rate
                    
                    # Handle subtitles
                    if config.write_subtitles:
                        if config.subtitles_word_level:
                            for token in data.tokens:
                                subs[word_idx] = {
                                    'text': token.text,
                                    'start': token.start_ts + cumulative_time,
                                    'end': token.end_ts + cumulative_time
                                }
                                word_idx += 1
                        else:
                            subs[i] = {
                                'text': data.graphemes,
                                'start': data.tokens[0].start_ts + cumulative_time,
                                'end': data.tokens[-1].end_ts + cumulative_time
                            }
                    
                    # Save chunk
                    chunk_path = Path(temp_dir) / f'{i}.wav'
                    sf.write(str(chunk_path), data.audio, config.sample_rate)
                    audio_chunks.append(chunk_path)
                    cumulative_time += chunk_duration
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk {i}: {e}")
            
            # Combine audio chunks
            combined_audio = []
            for chunk_path in audio_chunks:
                try:
                    audio, _ = sf.read(str(chunk_path))
                    
                    # Convert stereo to mono if needed
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    
                    if audio.size > 0:
                        combined_audio.append(audio)
                except Exception as e:
                    logger.error(f"Error reading audio chunk {chunk_path}: {e}")
            
            if combined_audio:
                try:
                    combined_audio = np.concatenate(combined_audio, axis=0)
                    
                    # Ensure output directory exists
                    output_path = Path(config.output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    sf.write(str(output_path), combined_audio, config.sample_rate)
                except Exception as e:
                    logger.error(f"Error writing combined audio: {e}")
                    raise KokoroException(f"Failed to write audio: {e}")
            else:
                raise KokoroException("No audio chunks generated")
        
        # Write subtitles
        if config.write_subtitles:
            self._write_srt_subtitles(subs, config.subtitles_path)
    
    @staticmethod
    def _write_srt_subtitles(subtitles: Dict[int, Dict], output_path: str):
        """Write subtitles in SRT format."""
        def srt_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, sub in subtitles.items():
                    f.write(f"{i+1}\n")
                    f.write(f"{srt_time(sub['start'])} --> {srt_time(sub['end'])}\n")
                    f.write(f"{sub['text']}\n\n")
        except Exception as e:
            logger.error(f"Error writing subtitles: {e}")
            raise KokoroException(f"Failed to write subtitles: {e}")


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='simpler_kokoro',
        description="SimplerKokoro CLI - Generate speech and list voices using Kokoro models."
    )
    parser.add_argument('--repo', type=str, default=SimplerKokoro.DEFAULT_REPO, 
                       help="HuggingFace repo to use for models.")
    parser.add_argument('--models_dir', type=str, default=SimplerKokoro.DEFAULT_MODELS_DIR,
                       help="Directory to store model files.")
    parser.add_argument('--log_level', type=str, default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help="Logging level.")
    parser.add_argument('--device', type=str, default="cpu",
                       help="Device to use for inference.")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # List voices command
    parser_list = subparsers.add_parser('list-voices', help='List available Kokoro voices.')
    parser_list.add_argument('--remote', action='store_true',
                            help='List voices from remote repository only.')
    parser_list.add_argument('--format', type=str, default='table',
                            choices=['table', 'json', 'names'],
                            help='Output format.')
    
    # Generate speech command
    parser_gen = subparsers.add_parser('generate', help='Generate speech audio from text.')
    parser_gen.add_argument('--text', type=str, required=True, help='Text to synthesize.')
    parser_gen.add_argument('--voice', type=str, required=True, help='Voice name to use.')
    parser_gen.add_argument('--output', type=str, required=True, help='Output WAV file path.')
    parser_gen.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier.')
    parser_gen.add_argument('--write_subtitles', action='store_true', help='Write SRT subtitles.')
    parser_gen.add_argument('--subtitles_path', type=str, default='subtitles.srt',
                           help='Path to save subtitles.')
    parser_gen.add_argument('--subtitles_word_level', action='store_true',
                           help='Word-level subtitles.')
    parser_gen.add_argument('--split_pattern', type=str, default=r'\.\s+|\n',
                           help='Regex pattern for splitting text.')
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    
    try:
        if args.command == 'list-voices':
            if args.remote:
                voices = SimplerKokoro.list_voices_remote(args.repo)
            else:
                sk = SimplerKokoro(
                    models_dir=args.models_dir,
                    repo=args.repo,
                    log_level=log_level,
                    device=args.device
                )
                voices = sk.voices
            
            if not voices:
                print("No voices found.")
                return
            
            if args.format == 'json':
                import json
                voice_list = [
                    {
                        'name': v.name,
                        'display_name': v.display_name,
                        'gender': v.gender,
                        'lang_code': v.lang_code
                    }
                    for v in voices
                ]
                print(json.dumps(voice_list, indent=2))
            elif args.format == 'names':
                for v in voices:
                    print(v.name)
            else:  # table
                print(f"{'Name':<20} {'Display Name':<20} {'Gender':<10} {'Lang':<6}")
                print('-' * 60)
                for v in voices:
                    print(f"{v.name:<20} {v.display_name:<20} {v.gender:<10} {v.lang_code:<6}")
        
        elif args.command == 'generate':
            sk = SimplerKokoro(
                models_dir=args.models_dir,
                repo=args.repo,
                log_level=log_level,
                device=args.device
            )
            
            sk.generate(
                text=args.text,
                voice=args.voice,
                output_path=args.output,
                speed=args.speed,
                write_subtitles=args.write_subtitles,
                subtitles_path=args.subtitles_path,
                subtitles_word_level=args.subtitles_word_level,
                split_pattern=args.split_pattern
            )
            
            print(f"✓ Audio saved to {args.output}")
            if args.write_subtitles:
                print(f"✓ Subtitles saved to {args.subtitles_path}")
    
    except KokoroException as e:
        logger.error(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
    