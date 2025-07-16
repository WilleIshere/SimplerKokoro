
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

---

## ✨ Features

- **Simple interface** for generating speech audio and subtitles
- **Supports all Kokoro voices**
- **Outputs valid SRT subtitles**
- **Automatic Model Management**

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

## 📦 Requirements

- Python 3.10+
- torch
- kokoro
- soundfile

<sub>All dependencies except Python are installed automatically.</sub>


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
    voice=voices[0]['name'],
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
