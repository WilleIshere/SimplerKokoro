# Simpler Kokoro - A simplified interface for generating speech and subtitles using Kokoro voices

import os
import warnings
import tempfile
import soundfile as sf
import huggingface_hub as hf
import logging


# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logger()


class SimplerKokoro:
    """
    SimplerKokoro provides a simplified interface for generating speech and subtitles using Kokoro voices.
    """
    def __init__(self, 
            device: str = "cpu",
            models_dir: str = 'models',
            repo: str = "hexgrad/Kokoro-82M",
            log_level: int = logging.INFO,
            skip_download: bool = False
        ):
        """
        Initialize SimplerKokoro.
        Args:
            device (str): Device to use for inference (default: "cpu").
            models_dir (str): Directory to store model files (default: 'models' in active directory).
            repo (str): HuggingFace repo to use for models (default: 'hexgrad/Kokoro-82M').
            log_level (int): Logging level (default: logging.INFO).
            skip_download (bool): If True, do not download models or create directories (default: False).
        """
        global logger
        logger = setup_logger(log_level)
        self.device = device
        self.models_dir = models_dir
        self.repo = repo
        self.kokoro_model_dir = os.path.join(self.models_dir, 'kokoro')
        self.kokoro_model_path = os.path.join(self.models_dir, 'kokoro', 'kokoro-v1_0.pth')
        self.kokoro_voices_path = os.path.join(self.models_dir, 'voices')
        if not skip_download:
            self.ensure_models_dirs()
            self.download_models()
            import kokoro
            self.kokoro = kokoro
            self.voices = self.list_voices()

    @staticmethod
    def list_voices_remote(repo: str) -> list[dict]:
        """
        Return a list of available Kokoro voices with metadata from HuggingFace only (no local files).
        Args:
            repo (str): HuggingFace repo to use for models.
        Returns:
            List[dict]: List of voice metadata dicts.
        """
        try:
            repo_files = hf.list_repo_files(repo)
            voice_files = [f for f in repo_files if f.startswith("voices/") and f.endswith(".pt")]
            voices = []
            for vf in voice_files:
                try:
                    voice = vf.lstrip('voices/').rstrip('.pt')
                    name = voice
                    display_name = voice[3:].capitalize()
                    lang_code = voice[0]
                    gender = 'Male' if voice[1] == 'm' else 'Female'
                    voices.append({
                        'name': name,
                        'display_name': display_name,
                        'gender': gender,
                        'lang_code': lang_code,
                        'model_path': vf  # Only remote path
                    })
                except Exception as e:
                    logger.error(f"Error parsing voice file {vf}: {e}")
            return voices
        except Exception as e:
            logger.error(f"Error fetching voice list from HuggingFace Hub: {e}")
            return []
        
    def download_models(self):
        """
        Download the Kokoro model files if they do not exist.
        Downloads the main model and voice files to the specified models directory.
        """
        try:
            if not os.path.exists(self.kokoro_model_path):
                logger.info("Downloading Main Kokoro model...")
                try:
                    hf.hf_hub_download(
                        repo_id=self.repo,
                        filename="kokoro-v1_0.pth",
                        local_dir=os.path.dirname(self.kokoro_model_path),
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.error(f"Error downloading main Kokoro model: {e}")
            try:
                repo_files = hf.list_repo_files(self.repo)
            except Exception as e:
                logger.error(f"Error fetching voice file list from HuggingFace Hub: {e}")
                return
            for voices_hf in repo_files:
                if voices_hf.startswith('voices/') and voices_hf.endswith('.pt'):
                    # voices_hf is like 'voices/xx_xxxxx.pt'
                    voice_filename = os.path.basename(voices_hf)
                    voice_file = os.path.join(self.kokoro_voices_path, voice_filename)
                    if not os.path.exists(voice_file):
                        logger.info(f"Downloading voice model: {voices_hf}")
                        try:
                            hf.hf_hub_download(
                                repo_id=self.repo,
                                filename=voices_hf,
                                local_dir=self.models_dir,
                                local_dir_use_symlinks=False
                            )
                        except Exception as e:
                            logger.error(f"Error downloading voice model {voices_hf}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in download_models: {e}")
            
        
    
    def ensure_models_dirs(self):
        """
        Ensure the necessary model directories exist.
        Creates the kokoro model directory and voices directory if they do not exist.
        """
        os.makedirs(self.kokoro_model_dir, exist_ok=True)
        os.makedirs(self.kokoro_voices_path, exist_ok=True)

    def generate(
        self,
        text: str,
        voice: str,
        output_path: str,
        speed: float = 1.0,
        write_subtitles: bool = False,
        subtitles_path: str = 'subtitles.srt',
        subtititles_word_level: bool = False
    ):
        """
        Generate speech audio and optional subtitles from text using a Kokoro voice.

        Args:
            text (str): The input text to synthesize.
            voice (str): The Kokoro voice name (e.g., 'af_alloy').
            output_path (str): Path to save the combined output audio file.
            speed (float): Speech speed multiplier (default: 1.0).
            write_subtitles (bool): Whether to write subtitles (default: False).
            subtitles_path (str): Path to save subtitles (default: 'subtitles.srt').
            subtititles_word_level (bool): If True, subtitles are word-level; else, chunk-level.
        """
        try:
            # Find the voice index and language code
            voice_index = next((i for i, v in enumerate(self.voices) if v['name'] == voice), 0)
            lang_code = self.voices[voice_index]['lang_code']
            model_path = self.voices[voice_index]['model_path']

            # Create Kokoro pipeline
            pipeline = self.kokoro.KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M"
            )

            # Use custom model if provided
            if model_path:
                try:
                    import torch
                    voice_model = torch.load(model_path, weights_only=True)
                    generator = pipeline(
                        text=text,
                        voice=voice_model,
                        speed=speed,
                        split_pattern=r'\.\s+|\n',
                    )
                except Exception as e:
                    logger.error(f"Error loading custom model: {e}")
                    logger.info("Falling back to default voice generation.")
                    generator = pipeline(
                        text=text,
                        voice=voice,
                        speed=speed,
                        split_pattern=r'\.\s+|\n',
                    )
            else:
                logger.info("Using default voice generation.")
                generator = pipeline(
                    text=text,
                    voice=voice,
                    speed=speed,
                    split_pattern=r'\.\s+|\n',
                )

            subs = {}
            word = 0
            audio_chunks = []
            cumulative_time = 0.0

            # Use a temporary directory for chunk files
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, data in enumerate(generator):
                    try:
                        chunk_duration = len(data.audio) / 24000  # samples / sample_rate
                        # Subtitle handling
                        if write_subtitles:
                            if subtititles_word_level:
                                for token in data.tokens:
                                    sub = {
                                        'text': token.text,
                                        'start': token.start_ts + cumulative_time,
                                        'end': token.end_ts + cumulative_time
                                    }
                                    subs[word] = sub
                                    word += 1
                            else:
                                start = data.tokens[0].start_ts + cumulative_time
                                end = data.tokens[-1].end_ts + cumulative_time
                                sub = {
                                    'text': data.graphemes,
                                    'start': start,
                                    'end': end
                                }
                                subs[i] = sub
                        # Write chunk to temp file
                        chunk_output_path = os.path.join(temp_dir, f'{i}.wav')
                        sf.write(chunk_output_path, data.audio, 24000)
                        audio_chunks.append(chunk_output_path)
                        cumulative_time += chunk_duration
                    except Exception as e:
                        logger.error(f"Error processing audio chunk {i}: {e}")

                # Combine all audio chunks
                import numpy as np
                combined_audio = []
                for chunk in audio_chunks:
                    try:
                        audio, samplerate = sf.read(chunk)
                        # Convert stereo to mono if needed
                        if audio.ndim > 1:
                            audio = audio.mean(axis=1)
                        if audio.size > 0:
                            combined_audio.append(audio)
                    except Exception as e:
                        logger.error(f"Error reading audio chunk {chunk}: {e}")
                if combined_audio:
                    try:
                        combined_audio = np.concatenate(combined_audio, axis=0)
                        sf.write(output_path, combined_audio, 24000)
                    except Exception as e:
                        logger.error(f"Error writing combined audio to {output_path}: {e}")
                else:
                    logger.warning("No audio chunks to combine.")

            # Write subtitles in SRT format
            if write_subtitles:
                def srt_time(seconds: float) -> str:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds - int(seconds)) * 1000)
                    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

                try:
                    with open(subtitles_path, 'w', encoding='utf-8') as f:
                        for i, sub in subs.items():
                            f.write(f"{i+1}\n")
                            f.write(f"{srt_time(sub['start'])} --> {srt_time(sub['end'])}\n")
                            f.write(f"{sub['text']}\n\n")
                except Exception as e:
                    logger.error(f"Error writing subtitles to {subtitles_path}: {e}")
        except Exception as e:
            logger.error(f"Error in generate: {e}")
    
    def list_voices(self) -> list[dict]:
        """
        Return a list of available Kokoro voices with metadata (local model_path).
        """
        try:
            repo_files = hf.list_repo_files(self.repo)
            voice_files = [f for f in repo_files if f.startswith("voices/") and f.endswith(".pt")]
            voices = []
            for vf in voice_files:
                try:
                    voice = vf.lstrip('voices/').rstrip('.pt')
                    name = voice
                    display_name = voice[3:].capitalize()
                    lang_code = voice[0]
                    gender = 'Male' if voice[1] == 'm' else 'Female'
                    voices.append({
                        'name': name,
                        'display_name': display_name,
                        'gender': gender,
                        'lang_code': lang_code,
                        'model_path': os.path.join(self.kokoro_voices_path, f"{voice}.pt")
                    })
                except Exception as e:
                    logger.error(f"Error parsing voice file {vf}: {e}")
            return voices
        except Exception as e:
            logger.error(f"Error fetching voice list from HuggingFace Hub: {e}")
            return []

# CLI interface
import argparse
def main():
    parser = argparse.ArgumentParser(prog='simpler_kokoro', description="SimplerKokoro CLI - Generate speech and list voices using Kokoro models.")
    parser.add_argument('--repo', type=str, default="hexgrad/Kokoro-82M", help="HuggingFace repo to use for models.")
    parser.add_argument('--models_dir', type=str, default="models", help="Directory to store model files.")
    parser.add_argument('--log_level', type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # List voices
    parser_list = subparsers.add_parser('list-voices', help='List available Kokoro voices.')

    # Generate speech
    parser_gen = subparsers.add_parser('generate', help='Generate speech audio from text.')
    parser_gen.add_argument('--text', type=str, required=True, help='Text to synthesize.')
    parser_gen.add_argument('--voice', type=str, required=True, help='Voice name to use.')
    parser_gen.add_argument('--output', type=str, required=True, help='Output WAV file path.')
    parser_gen.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier.')
    parser_gen.add_argument('--write_subtitles', action='store_true', help='Write SRT subtitles.')
    parser_gen.add_argument('--subtitles_path', type=str, default='subtitles.srt', help='Path to save subtitles.')
    parser_gen.add_argument('--subtitles_word_level', action='store_true', help='Word-level subtitles.')

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    if args.command == 'list-voices':
        voices = SimplerKokoro.list_voices_remote(args.repo)
        if not voices:
            print("No voices found.")
        else:
            print(f"{'Name':20} {'Display Name':20} {'Gender':8} {'Lang':6}")
            print('-'*60)
            for v in voices:
                print(f"{v['name']:20} {v['display_name']:20} {v['gender']:8} {v['lang_code']:6}")
    elif args.command == 'generate':
        sk = SimplerKokoro(models_dir=args.models_dir, repo=args.repo, log_level=log_level)
        sk.generate(
            text=args.text,
            voice=args.voice,
            output_path=args.output,
            speed=args.speed,
            write_subtitles=args.write_subtitles,
            subtitles_path=args.subtitles_path,
            subtititles_word_level=args.subtitles_word_level
        )
        print(f"Audio saved to {args.output}")
        if args.write_subtitles:
            print(f"Subtitles saved to {args.subtitles_path}")
