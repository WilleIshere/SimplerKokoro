# SimplerKokoro

SimplerKokoro is a Python package that makes it easy to use the Kokoro voice synthesis library.

## Features
- Simple interface for generating speech audio and subtitles
- Supports all Kokoro voices
- Outputs valid SRT subtitles
- Automatic Model Management

## Installation

Install from PyPI:

```bash
pip install Simpler-Kokoro
```

or clone the repo and install locally:

```bash
git clone https://github.com/WilleIshere/SimplerKokoro.git
cd SimplerKokoro
pip install .
```

## Requirements
- Python 3.10+
- torch
- kokoro
- soundfile

All dependencies are installed automatically.

## Usage

### Basic Example

```python
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
```

### Generate Speech with Subtitles

```python
sk.generate(
    text="Hello, this is a test. This is another sentence.",
    voice=voices[0]['name'],
    output_path="output.wav",
    write_subtitles=True,
    subtitles_path="output.srt",
    subtititles_word_level=True
)
```

### Generate Speech with Custom Speed

```python
sk.generate(
    text="This is spoken faster than normal.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
    speed=1.5
)
```

### Example Output Files

- `output.wav`: The synthesized speech audio file.
- `output.srt`: Subtitles in SRT format (if `write_subtitles=True`).

Sample SRT output:
```
1
00:00:00,000 --> 00:00:01,200
Hello,

2
00:00:01,200 --> 00:00:02,500
this is a test.

3
00:00:02,500 --> 00:00:04,000
This is another sentence.
```

## API

### SimplerKokoro

#### Methods
- `list_voices()`: Returns a list of available voices with metadata.
- `generate(text, voice, output_path, speed=1.0, write_subtitles=False, subtitles_path='subtitles.srt', subtititles_word_level=False)`: Generates speech audio and optional subtitles.

## License

GPL-3.0 license
