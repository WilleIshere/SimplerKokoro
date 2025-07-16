from Simpler_Kokoro import SimplerKokoro

sk = SimplerKokoro()
voices = sk.list_voices()

sk.generate(
    models_dir="Folder-to-put-models-in",
    text="Thats a cool model directory.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
)
