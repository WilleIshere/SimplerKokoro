from Simpler_Kokoro import SimplerKokoro

sk = SimplerKokoro()
voices = sk.list_voices()

sk.generate(
    text="This is spoken faster than normal.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
    speed=1.5
)
