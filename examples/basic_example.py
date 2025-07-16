from Simpler_Kokoro import SimplerKokoro

# Create an instance
sk = SimplerKokoro()

# List available voices
voices = sk.list_voices()
print("Available voices:", [v['name'] for v in voices])

# Generate speech
sk.generate(
    text="Hello, this is a test of the Simpler Kokoro voice synthesis.",
    voice=voices[0]['name'],
    output_path="output.wav"
)
