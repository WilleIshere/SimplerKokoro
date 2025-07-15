import pytest
from Simpler_Kokoro.Simpler_Kokoro import SimplerKokoro

from pprint import pprint

def test_load_pipeline():
    SimplerKokoro()

def test_list_voices():
    simpler_kokoro = SimplerKokoro()
    simpler_kokoro.list_voices()
    
def test_generate():
    simpler_kokoro = SimplerKokoro()
    voices = simpler_kokoro.list_voices()
    voice = voices[0]
    
    output_path = "output.wav"
    
    simpler_kokoro.generate(
        text="Hello, this is a test of the Simpler Kokoro voice synthesis.",
        voice=voice,
        output_path=output_path
    )
    
    import os
    assert os.path.exists(output_path), "Output file was not created."


if __name__ == "__main__":
    simpler_kokoro = SimplerKokoro()
    voices = simpler_kokoro.list_voices()
    voice = voices[0]['name']
    
    simpler_kokoro.generate(
        text="Hello, this is a test of the Simpler Kokoro voice synthesis. Hello, this is a test of the Simpler Kokoro voice synthesis. Hello, this is a test of the Simpler Kokoro voice synthesis.\n Hello, this is a test of the Simpler Kokoro voice synthesis.",
        voice=voice,
        output_path="output.wav",
        write_subtitles=True,
        subtititles_word_level=True
        )