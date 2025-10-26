from Simpler_Kokoro import SimplerKokoro

# Create an instance
sk = SimplerKokoro()

# Load the available voices
voices = sk.list_voices()

# Use the first voice as example
selected_voice = voices[0]

# Generate speech
sk.generate(
    text='Hello, this is a test of the Simpler Kokoro voice synthesis.', # Text to generate 
    voice=selected_voice.name, # Grab the name from the selected voice
    output_path='output.wav', # Select the output path
    speed=1.5 # This represents 150% Speed. 1 means 100% and 0.5 means 50%
)