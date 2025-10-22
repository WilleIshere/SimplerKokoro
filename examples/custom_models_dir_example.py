from Simpler_Kokoro import SimplerKokoro

sk = SimplerKokoro(models_dir="Folder-to-put-models-in",) # Here is the models_dir parameter
voices = sk.list_voices()

sk.generate(
    text="Thats a cool model directory.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
)
