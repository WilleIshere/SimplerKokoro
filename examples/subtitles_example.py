from Simpler_Kokoro import SimplerKokoro

sk = SimplerKokoro()
voices = sk.list_voices()

sk.generate(
    text="Hello, this is a test. This is another sentence.",
    voice=voices[0]['name'],
    output_path="output.wav",
    write_subtitles=True,
    subtitles_path="output.srt",
    subtititles_word_level=True
)
