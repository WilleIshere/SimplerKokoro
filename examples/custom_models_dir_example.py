from Simpler_Kokoro import SimplerKokoro

# Create an instance
sk = SimplerKokoro(models_dir='<PATH TO PUT MODELS>') # Put in the path where you want the models to be saved here

# Load the available voices
voices = sk.list_voices()

# Use the first voice as example
selected_voice = voices[0]

# Generate speech
sk.generate(
    text='Select a custom directory for the models!', # Text to generate 
    voice=selected_voice.name, # Grab the name from the selected voice
    output_path='output.wav' # Select the output path.
)
