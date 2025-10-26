from Simpler_Kokoro import SimplerKokoro

# Create an instance
sk = SimplerKokoro()

# Load the available voices
voices = sk.list_voices()

# Use the first voice as example
selected_voice = voices[0]

# Generate speech
sk.generate(
    text='Hello, this will generate a subtitles.srt file along with output.wav', # Text to generate
    voice=selected_voice.name, # Grab the name from the selected voice
    output_path='output.wav', # Select the output path
    write_subtitles=True, # Enable subtitle generation
    subtitles_path='subtitles.srt', # (optional) Specify the subtitle .srt filename
    subtitles_word_level=True # (optional) Enable word level timestamps
)