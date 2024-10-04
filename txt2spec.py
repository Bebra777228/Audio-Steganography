import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import librosa
import librosa.display
import gradio as gr
import soundfile as sf
import os
import logging
import tempfile

# Constants
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEFAULT_SAMPLE_RATE = 22050

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function for creating a spectrogram image with text
def text_to_spectrogram_image(text, base_width=512, height=256, max_font_size=80, margin=10, letter_spacing=5):
    try:
        font = ImageFont.truetype(DEFAULT_FONT_PATH, max_font_size)
    except IOError:
        logging.warning(f"Font not found at {DEFAULT_FONT_PATH}. Using default font.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(Image.new('L', (1, 1)))

    text_width = sum(draw.textbbox((0, 0), char, font=font)[2] - draw.textbbox((0, 0), char, font=font)[0] + letter_spacing for char in text) - letter_spacing
    text_height = draw.textbbox((0, 0), text[0], font=font)[3] - draw.textbbox((0, 0), text[0], font=font)[1]

    # Adjust width and height based on text size
    width = max(base_width, text_width + margin * 2)
    height = max(height, text_height + margin * 2)

    image = Image.new('L', (width, height), 'black')
    draw = ImageDraw.Draw(image)

    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    for char in text:
        draw.text((text_x, text_y), char, font=font, fill='white')
        char_bbox = draw.textbbox((0, 0), char, font=font)
        text_x += char_bbox[2] - char_bbox[0] + letter_spacing

    image = np.array(image)
    image = np.where(image > 0, 255, image)
    return image

# Function for creating a spectrogram image with an image
def image_to_spectrogram_image(image_path, base_width=512, height=256, margin=10):
    image = Image.open(image_path).convert('L')
    image_width, image_height = image.size

    # Adjust width and height based on image size
    width = max(base_width, image_width + margin * 2)
    height = max(height, image_height + margin * 2)

    new_image = Image.new('L', (width, height), 'black')
    new_image.paste(image, ((width - image_width) // 2, (height - image_height) // 2))

    new_image = np.array(new_image)
    new_image = np.where(new_image > 0, 255, new_image)
    return new_image

# Converting an image to audio
def spectrogram_image_to_audio(image, sr=DEFAULT_SAMPLE_RATE):
    flipped_image = np.flipud(image)
    S = flipped_image.astype(np.float32) / 255.0 * 100.0
    y = librosa.griffinlim(S)
    return y

# Function for creating an audio file and spectrogram from text
def create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing):
    spec_image = text_to_spectrogram_image(text, base_width, height, max_font_size, margin, letter_spacing)
    y = spectrogram_image_to_audio(spec_image)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_path = temp_audio.name
        sf.write(audio_path, y, DEFAULT_SAMPLE_RATE)
    
    # Create spectrogram from audio
    S = librosa.feature.melspectrogram(y=y, sr=DEFAULT_SAMPLE_RATE)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=DEFAULT_SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
    return audio_path, spectrogram_path

# Function for creating an audio file and spectrogram from an image
def create_audio_with_spectrogram_from_image(image_path, base_width, height, margin):
    spec_image = image_to_spectrogram_image(image_path, base_width, height, margin)
    y = spectrogram_image_to_audio(spec_image)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_path = temp_audio.name
        sf.write(audio_path, y, DEFAULT_SAMPLE_RATE)
    
    # Create spectrogram from audio
    S = librosa.feature.melspectrogram(y=y, sr=DEFAULT_SAMPLE_RATE)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=DEFAULT_SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
    return audio_path, spectrogram_path

# Function for displaying the spectrogram of an audio file
def display_audio_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    return spectrogram_path

# Gradio interface
def gradio_interface_fn(text, base_width, height, max_font_size, margin, letter_spacing):
    logging.info(f"Generating audio and spectrogram for text: {text}")
    audio_path, spectrogram_path = create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing)
    return audio_path, spectrogram_path

def gradio_image_to_audio_fn(upload_image, base_width, height, margin):
    logging.info(f"Converting image to audio: {upload_image}")
    audio_path, spectrogram_path = create_audio_with_spectrogram_from_image(upload_image, base_width, height, margin)
    return audio_path, spectrogram_path

def gradio_decode_fn(upload_audio):
    logging.info(f"Generating spectrogram for audio: {upload_audio}")
    return display_audio_spectrogram(upload_audio)

with gr.Blocks(title='Audio Steganography', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", spacing_size="sm", radius_size="lg")) as txt2spec:
    with gr.Tab("Text to Spectrogram"):
        with gr.Group():
            text = gr.Textbox(lines=2, placeholder="Enter your text:", label="Text")
            with gr.Row(variant='panel'):
                base_width = gr.Slider(value=512, label="Image Width", visible=False)
                height = gr.Slider(value=256, label="Image Height", visible=False)
                max_font_size = gr.Slider(minimum=10, maximum=130, step=5, value=80, label="Font size")
                margin = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Indent")
                letter_spacing = gr.Slider(minimum=0, maximum=50, step=1, value=5, label="Letter spacing")
            generate_button = gr.Button("Generate")

        with gr.Column(variant='panel'):
            with gr.Group():
                output_audio = gr.Audio(type="filepath", label="Generated audio")
                output_spectrogram = gr.Image(type="filepath", label="Spectrogram")

        generate_button.click(gradio_interface_fn, inputs=[text, base_width, height, max_font_size, margin, letter_spacing], outputs=[output_audio, output_spectrogram])

    with gr.Tab("Image to Spectrogram"):
        with gr.Group():
            with gr.Row(variant='panel'):
                upload_image = gr.Image(type="filepath", label="Upload image")
                base_width = gr.Slider(value=512, label="Image Width", visible=False)
                height = gr.Slider(value=256, label="Image Height", visible=False)
                margin = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Indent")
                convert_button = gr.Button("Convert to audio")

            with gr.Column(variant='panel'):
                output_audio_from_image = gr.Audio(type="filepath", label="Generated audio")
                output_spectrogram_from_image = gr.Image(type="filepath", label="Spectrogram")

            convert_button.click(gradio_image_to_audio_fn, inputs=[upload_image, base_width, height, margin], outputs=[output_audio_from_image, output_spectrogram_from_image])

    with gr.Tab("Audio Spectrogram"):
        with gr.Group():
            with gr.Row(variant='panel'):
                upload_audio = gr.Audio(type="filepath", label="Upload audio", scale=3)
                decode_button = gr.Button("Show spectrogram", scale=2)

            with gr.Column(variant='panel'):
                decoded_image = gr.Image(type="filepath", label="Audio Spectrogram")

            decode_button.click(gradio_decode_fn, inputs=[upload_audio], outputs=[decoded_image])

txt2spec.launch(share=True)
