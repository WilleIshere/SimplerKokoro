# Simpler Kokoro - A simplified interface for generating speech and subtitles using Kokoro voices
import os
import warnings
import tempfile
import soundfile as sf
import huggingface_hub as hf

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SimplerKokoro:
    """
    SimplerKokoro provides a simplified interface for generating speech and subtitles using Kokoro voices.
    """
    def __init__(self, 
            device: str = "cpu",
            models_dir: str = 'models'
        ):
        """
        Initialize SimplerKokoro.
        Args:
            device (str): Device to use for inference (default: "cpu").
            models_dir (str): Directory to store model files (default: 'models' in active directory).
        """
        self.device = device
        
        self.models_dir = models_dir
        
        self.kororo_model_path = os.path.join(self.models_dir, 'kokoro')
        self.kokoro_voices_path = os.path.join(self.models_dir, 'voices')
        
        self.kokoro_model_path = os.path.join(self.models_dir, 'kokoro', 'kokoro-v1_0.pth')
        
        self.ensure_models_dirs()
        self.download_models()
        
        import kokoro
        self.kokoro = kokoro
        
        self.voices = self.list_voices()
        
    def download_models(self):
        """
        Download the Kokoro model files if they do not exist.
        Downloads the main model and voice files to the specified models directory.
        """
        try:
            if not os.path.exists(self.kokoro_model_path):
                print("Downloading Main Kokoro model...")
                try:
                    hf.hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename="kokoro-v1_0.pth",
                        local_dir=self.kororo_model_path,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"Error downloading main Kokoro model: {e}")
            try:
                repo_files = hf.list_repo_files("hexgrad/Kokoro-82M")
            except Exception as e:
                print(f"Error fetching voice file list from HuggingFace Hub: {e}")
                return
            for voices_hf in repo_files:
                if voices_hf.startswith('voices/') and voices_hf.endswith('.pt'):
                    voice_file = os.path.join(self.kokoro_voices_path, voices_hf)
                    if not os.path.exists(voice_file):
                        print(f"Downloading voice model: {voices_hf}")
                        try:
                            hf.hf_hub_download(
                                repo_id="hexgrad/Kokoro-82M",
                                filename=voices_hf,
                                local_dir=self.models_dir,
                                local_dir_use_symlinks=False
                            )
                        except Exception as e:
                            print(f"Error downloading voice model {voices_hf}: {e}")
        except Exception as e:
            print(f"Unexpected error in download_models: {e}")
            
        
    
    def ensure_models_dirs(self):
        """
        Ensure the necessary model directories exist.
        Creates the kokoro model directory and voices directory if they do not exist.
        """
        os.makedirs(self.kororo_model_path, exist_ok=True)
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
                    print(f"Error loading custom model: {e}")
                    print("Falling back to default voice generation.")
                    generator = pipeline(
                        text=text,
                        voice=voice,
                        speed=speed,
                        split_pattern=r'\.\s+|\n',
                    )
            else:
                print("Using default voice generation.")
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
                        print(f"Error processing audio chunk {i}: {e}")

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
                        print(f"Error reading audio chunk {chunk}: {e}")
                if combined_audio:
                    try:
                        combined_audio = np.concatenate(combined_audio, axis=0)
                        sf.write(output_path, combined_audio, 24000)
                    except Exception as e:
                        print(f"Error writing combined audio to {output_path}: {e}")
                else:
                    print("No audio chunks to combine.")

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
                    print(f"Error writing subtitles to {subtitles_path}: {e}")
        except Exception as e:
            print(f"Error in generate: {e}")
    
    def list_voices(self) -> list[dict]:
        """
        Return a list of available Kokoro voices with metadata.
        Returns:
            List[dict]: List of voice metadata dicts.
        Raises:
            RuntimeError: If unable to fetch voice list from HuggingFace Hub.
        """
        try:
            repo_files = hf.list_repo_files("hexgrad/Kokoro-82M")
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
                    print(f"Error parsing voice file {vf}: {e}")
            return voices
        except Exception as e:
            print(f"Error fetching voice list from HuggingFace Hub: {e}")
            return []
