import kokoro
import os

class SimplerKokoro:
    def __init__(self,
            lang_code: str = "a",
            kokoro_model_path: str = None,
            voice_model_path: str = None,
            device: str = "cpu"  # Default to CPU, can be changed to "cuda" for GPU support
        ):
        
        self.lang_code = lang_code
        self.kokoro_model_path = kokoro_model_path
        self.voice_model_path = voice_model_path
        self.device = device
        
        self.pipeline = None
        
    def load_pipeline(self):
        from kokoro import KPipeline
        self.pipeline = KPipeline(
            lang_code=self.lang_code,
        )