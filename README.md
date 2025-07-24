
<p align="center">
  <img src="https://github.com/WilleIshere/SimplerKokoro/blob/main/poster.jpg?raw=true" alt="SimplerKokoro" width="60%">
</p>

<h1 align="center">SimplerKokoro</h1>

<p align="center">
  <b>Effortless speech synthesis with Kokoro, in Python.</b><br>
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/simpler-kokoro">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/simpler-kokoro">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/WilleIshere/SimplerKokoro">
<img alt="PyPI - License" src="https://img.shields.io/pypi/l/simpler-kokoro">


</p>

<p align="center">
  <a href="https://pypi.org/project/Simpler-Kokoro/" style="font-size:1.1em;"><b>View on PyPI</b></a>
</p>

## 📚 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Examples](#-examples)
- [Usage](#-usage)
- [CLI](#-command-line-interface)
- [Example Output Files](#-example-output-files)
- [Build from Source](#-build-from-source)
- [API](#-api)
- [License](#-license)

---

## ✨ Features

- **Simple interface** for generating speech audio and subtitles
- **Supports all Kokoro voices**
- **Outputs valid SRT subtitles**
- **Automatic Model Management**

---

## 📦 Requirements

- Python 3.10+
- torch
- kokoro
- soundfile

<sub>All dependencies except Python are installed automatically.</sub>

---

## 🚀 Installation

**From PyPI:**

```bash
pip install Simpler-Kokoro
```

**Or clone the repo and install locally:**

```bash
git clone https://github.com/WilleIshere/SimplerKokoro.git
cd SimplerKokoro
pip install .
```

---

## 🧑‍💻 Examples

You can find runnable example scripts in the [`examples/`](examples) folder:

- [`basic_example.py`](examples/basic_example.py): Basic usage, generate speech from text.
- [`subtitles_example.py`](examples/subtitles_example.py): Generate speech with SRT subtitles.
- [`custom_speed_example.py`](examples/custom_speed_example.py): Generate speech with custom speed.
- [`custom_models_dir_example.py`](examples/custom_models_dir_example.py): Specify a custom directory for model downloads.

---

## 🛠️ Usage

<details>
<summary><b>Basic Example</b></summary>

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
    voice=voices[0]['name'], # Make sure to specify the name of the voice
    output_path="output.wav"
)
```
</details>

<details>
<summary><b>Generate Speech with Subtitles</b></summary>

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
</details>

<details>
<summary><b>Generate Speech with Custom Speed</b></summary>

```python
sk.generate(
    text="This is spoken faster than normal.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
    speed=1.5
)
```
</details>

<details>
<summary><b>Specify a Path to Download Models</b></summary>

```python
sk.generate(
    models_dir="Folder-to-put-models-in",
    text="Thats a cool model directory.",
    voice=voices[1]['name'],
    output_path="fast_output.wav",
)
```
</details>

---

## 🖥️ Command Line Interface (CLI)

You can use the library in the command line too.

Example:

```bash
python -m Simpler_Kokoro <command> [options]
```

#### Commands and Options

| Command         | Description                        | Options                                                                                       |
|-----------------|------------------------------------|-----------------------------------------------------------------------------------------------|
| list-voices     | List available Kokoro voices       | --repo, --models_dir, --log_level                                                            |
| generate        | Generate speech audio from text    | --text (required), --voice (required), --output (required), --speed, --write_subtitles, --subtitles_path, --subtitles_word_level, --repo, --models_dir, --log_level |

**Global options:**

| Option              | Description                                      | Default                |
|---------------------|--------------------------------------------------|------------------------|
| --repo              | HuggingFace repo to use for models               | hexgrad/Kokoro-82M     |
| --models_dir        | Directory to store model files                   | models                 |
| --log_level         | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO              |

**Generate command options:**

| Option                  | Description                                 | Default         |
|-------------------------|---------------------------------------------|-----------------|
| --text                  | Text to synthesize (required)               |                 |
| --voice                 | Voice name to use (required)                |                 |
| --output                | Output WAV file path (required)             |                 |
| --speed                 | Speech speed multiplier                     | 1.0             |
| --write_subtitles       | Write SRT subtitles                         | False           |
| --subtitles_path        | Path to save subtitles                      | subtitles.srt    |
| --subtitles_word_level  | Word-level subtitles                        | False           |


---

### 📂 Example Output Files

- `output.wav`: The synthesized speech audio file.
- `output.srt`: Subtitles in SRT format (if `write_subtitles=True`).

<details>
<summary>Sample SRT output</summary>

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
</details>

---

## 🏗️ Build from Source

To build the package from source:

```bash
git clone https://github.com/WilleIshere/SimplerKokoro.git
cd SimplerKokoro
pip install build
python -m build
```

This will create distribution files in the `dist/` directory:

- `.whl` (wheel) file for pip installation
- `.tar.gz` source archive

To install the built wheel locally:

```bash
pip install dist/Simpler_Kokoro-*.whl
```

You can now use the package as described in the usage section.


---


## 📖 API

### <code>SimplerKokoro</code>

#### Methods

- <code>list_voices()</code>: Returns a list of available voices with metadata.
- <code>generate(text, voice, output_path, speed=1.0, write_subtitles=False, subtitles_path='subtitles.srt', subtititles_word_level=False)</code>: Generates speech audio and optional subtitles.

---

## 📄 License

This project is licensed under the **GPL-3.0** license.

---

<h2 align="center">⭐ Star History</h2>

<p align="center">
  <a href="https://star-history.com/#WilleIshere/SimplerKokoro&Date">
    <img src="https://api.star-history.com/svg?repos=WilleIshere/SimplerKokoro&type=Date" alt="Star History Chart" width="60%" />
  </a>
</p>
