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

# List of available Kokoro voice files (borrowed from Kokoro-Local-Gui)
VOICE_FILES = [
    # American Female voices
    "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
    "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt",
    "af_sarah.pt", "af_sky.pt",
    # American Male voices
    "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt",
    "am_liam.pt", "am_michael.pt", "am_onyx.pt", "am_puck.pt",
    "am_santa.pt",
    # British Female voices
    "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",
    # British Male voices
    "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",
    # Special voices
    "el_dora.pt", "em_alex.pt", "em_santa.pt",
    "ff_siwis.pt",
    "hf_alpha.pt", "hf_beta.pt",
    "hm_omega.pt", "hm_psi.pt",
    "jf_sara.pt", "jm_nicola.pt",
    "jf_alpha.pt", "jf_gongtsuene.pt", "jf_nezumi.pt", "jf_tebukuro.pt",
    "jm_kumo.pt",
    "pf_dora.pt", "pm_alex.pt", "pm_santa.pt",
    "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoqiao.pt", "zf_xiaoyi.pt"
]



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
        if not os.path.exists(self.kokoro_model_path):
            hf.hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename="kokoro-v1_0.pth",
                local_dir=self.kororo_model_path,
                local_dir_use_symlinks=False
            )
            
        for voices_hf in hf.list_repo_files("hexgrad/Kokoro-82M"):
            if voices_hf.lstrip('voices/') in VOICE_FILES:
                voice_file = os.path.join(self.kokoro_voices_path, voices_hf)
                if not os.path.exists(voice_file):
                    hf.hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename=voices_hf,
                        local_dir=self.models_dir,
                        local_dir_use_symlinks=False
                    )
            
        
    
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

            # Combine all audio chunks
            import numpy as np
            combined_audio = []
            for chunk in audio_chunks:
                audio, samplerate = sf.read(chunk)
                # Convert stereo to mono if needed
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if audio.size > 0:
                    combined_audio.append(audio)
            if combined_audio:
                combined_audio = np.concatenate(combined_audio, axis=0)
                sf.write(output_path, combined_audio, 24000)
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

            with open(subtitles_path, 'w', encoding='utf-8') as f:
                for i, sub in subs.items():
                    f.write(f"{i+1}\n")
                    f.write(f"{srt_time(sub['start'])} --> {srt_time(sub['end'])}\n")
                    f.write(f"{sub['text']}\n\n")
    
    def list_voices(self):
        """
        Return a list of available Kokoro voices with metadata.
        Returns:
            List[dict]: List of voice metadata dicts.
        """
        voices_list = [x.replace('.pt', '') for x in VOICE_FILES]
        voices = []
        for voice in voices_list:
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
        return voices