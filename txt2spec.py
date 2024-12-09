import logging
import tempfile
import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEFAULT_SAMPLE_RATE = 22050

logging.basicConfig(level=logging.INFO)

def text_to_spectrogram_image(text, base_width=512, height=256, max_font_size=80, margin=10, letter_spacing=5):
    try:
        font = ImageFont.truetype(DEFAULT_FONT_PATH, max_font_size)
    except IOError:
        logging.warning(f"Font not found at {DEFAULT_FONT_PATH}. Using default font.")
        font = ImageFont.load_default()
    except Exception as e:
        logging.error(f"An error occurred while loading the font: {e}")
        raise

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    text_widths = [
        draw.textbbox((0, 0), char, font=font)[2] - draw.textbbox((0, 0), char, font=font)[0]
        for char in text
    ]
    text_width = sum(text_widths) + letter_spacing * (len(text) - 1)
    text_height = (
        draw.textbbox((0, 0), text[0], font=font)[3]
        - draw.textbbox((0, 0), text[0], font=font)[1]
    )

    width = max(base_width, text_width + margin * 2)
    height = max(height, text_height + margin * 2)

    image = Image.new("L", (width, height), "black")
    draw = ImageDraw.Draw(image)

    text_start_x = (width - text_width) // 2
    text_start_y = (height - text_height) // 2

    current_x = text_start_x
    for char, char_width in zip(text, text_widths):
        draw.text((current_x, text_start_y), char, font=font, fill="white")
        current_x += char_width + letter_spacing

    image = np.array(image)
    image = np.where(image > 0, 255, image)
    return image

def spectrogram_image_to_audio(image, sr=DEFAULT_SAMPLE_RATE):
    flipped_image = np.flipud(image)
    S = flipped_image.astype(np.float32) / 255.0 * 100.0
    y = librosa.griffinlim(S)
    return y

def create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing):
    spec_image = text_to_spectrogram_image(text, base_width, height, max_font_size, margin, letter_spacing)
    y = spectrogram_image_to_audio(spec_image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        sf.write(audio_path, y, DEFAULT_SAMPLE_RATE)

    S = librosa.feature.melspectrogram(y=y, sr=DEFAULT_SAMPLE_RATE)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=DEFAULT_SAMPLE_RATE, x_axis="time", y_axis="mel")
    plt.axis("off")
    plt.tight_layout(pad=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    return audio_path, spectrogram_path

def display_audio_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")
    plt.tight_layout(pad=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    return spectrogram_path

def image_to_spectrogram_audio(image_path, sr=DEFAULT_SAMPLE_RATE):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    y = spectrogram_image_to_audio(image, sr)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        img2audio_path = temp_audio.name
        sf.write(img2audio_path, y, sr)
    return img2audio_path

def gradio_interface_fn(text, base_width, height, max_font_size, margin, letter_spacing):
    audio_path, spectrogram_path = create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing)
    return audio_path, spectrogram_path

def gradio_image_to_audio_fn(upload_image):
    return image_to_spectrogram_audio(upload_image)

def gradio_decode_fn(upload_audio):
    return display_audio_spectrogram(upload_audio)

with gr.Blocks(title="Audio Steganography", css="footer{display:none !important}", theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", spacing_size="sm", radius_size="lg")) as txt2spec:
    with gr.Tab("Text to Spectrogram"):
        with gr.Group():
            text = gr.Textbox(lines=2, placeholder="Enter your text:", label="Text", info="Enter the text you want to convert to audio.")
            with gr.Row(variant="panel"):
                base_width = gr.Slider(value=512, label="Image Width", visible=False)
                height = gr.Slider(value=256, label="Image Height", visible=False)
                max_font_size = gr.Slider(minimum=10, maximum=130, step=5, value=80, label="Font size")
                margin = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Indent")
                letter_spacing = gr.Slider(minimum=0, maximum=50, step=1, value=5, label="Letter spacing")
            generate_button = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            with gr.Group():
                output_audio = gr.Audio(type="filepath", label="Generated audio")
                output_spectrogram = gr.Image(type="filepath", label="Spectrogram")

        generate_button.click(gradio_interface_fn, inputs=[text, base_width, height, max_font_size, margin, letter_spacing], outputs=[output_audio, output_spectrogram])

    with gr.Tab("Image to Spectrogram"):
        with gr.Group():
            with gr.Column():
                upload_image = gr.Image(type="filepath", label="Upload image")
                convert_button = gr.Button("Convert to audio", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            output_audio_from_image = gr.Audio(type="filepath", label="Generated audio")

        convert_button.click(gradio_image_to_audio_fn, inputs=[upload_image], outputs=[output_audio_from_image])

    with gr.Tab("Audio Spectrogram"):
        with gr.Group():
            with gr.Column():
                upload_audio = gr.Audio(type="filepath", label="Upload audio", scale=3)
                decode_button = gr.Button("Show spectrogram", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            decoded_image = gr.Image(type="filepath", label="Audio Spectrogram")

        decode_button.click(gradio_decode_fn, inputs=[upload_audio], outputs=[decoded_image])

txt2spec.launch(share=True)
