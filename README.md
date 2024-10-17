

# Audio Steganography

This repository provides a tool for creating audio steganography by converting text into spectrogram images and generating audio from these images. Additionally, it supports converting uploaded images into audio and displaying spectrograms for uploaded audio files. The project uses Python libraries such as `librosa`, `matplotlib`, `PIL`, and `Gradio` to provide an easy-to-use interface for these tasks.

## Features

- **Text to Audio & Spectrogram**: Convert text into a spectrogram image and generate corresponding audio.
- **Image to Audio**: Convert an uploaded image into audio.
- **Audio Spectrogram**: Display the spectrogram for uploaded audio files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bebra777228/Audio-Steganography.git
   cd Audio-Steganography
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To launch the Gradio interface, run:
```bash
python txt2spec.py
```

This will open a local Gradio interface where you can interact with the following tabs:

### 1. Text to Spectrogram
   - Enter text and adjust parameters such as image width, height, font size, margin, and letter spacing.
   - Click **Generate** to create an audio file and its spectrogram.

### 2. Image to Spectrogram
   - Upload an image and click **Convert to audio** to generate an audio file based on the spectrogram.

### 3. Audio Spectrogram
   - Upload an audio file and click **Show spectrogram** to display its spectrogram image.

## File Descriptions

- `txt2spec.py`: Main script containing functions for text-to-spectrogram and image-to-audio conversions, and Gradio interface setup.
  
## Dependencies

- Python 3.7+
- `numpy`
- `matplotlib`
- `Pillow`
- `librosa`
- `soundfile`
- `tempfile`
- `gradio`
- `os`
- `logging`

## Gradio Interface Components

The Gradio interface includes the following components:

- **Text to Spectrogram**: Converts text into audio and generates a spectrogram.
- **Image to Spectrogram**: Converts uploaded images to audio.
- **Audio Spectrogram**: Displays spectrograms of uploaded audio files.

## License

This project is open-source under the MIT License. See the [LICENSE](LICENSE) file for more details.

