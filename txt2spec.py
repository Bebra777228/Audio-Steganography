import logging
import os
import tempfile
import gradio as gr
import librosa
import matplotlib.cm as cm
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    os.path.expanduser("~/Library/Fonts/Druk Wide.otf"),
    os.path.expanduser("~/Library/Fonts/Druk Wide.ttf"),
    os.path.expanduser("~/Library/Fonts/DrukWide.otf"),
    os.path.expanduser("~/Library/Fonts/DrukWide.ttf"),
    "/Library/Fonts/Druk Wide.otf",
    "/Library/Fonts/Druk Wide.ttf",
    "/Library/Fonts/DrukWide.otf",
    "/Library/Fonts/DrukWide.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Helvetica.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def find_font_path():
    for path in FONT_PATHS:
        if os.path.exists(path):
            return path
    return None

DEFAULT_FONT_PATH = find_font_path()
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = DEFAULT_N_FFT // 4
DEFAULT_SPEC_HEIGHT = DEFAULT_N_FFT // 2 + 1
DEFAULT_BASE_WIDTH = 1024
DEFAULT_MARGIN = 10
DEFAULT_LETTER_SPACING = 5
DEFAULT_FONT_SIZE_CAP = DEFAULT_SPEC_HEIGHT
DEFAULT_FONT_SIZE = max(10, DEFAULT_SPEC_HEIGHT - DEFAULT_MARGIN * 2)
DEFAULT_FLUTTER_PX = 6
DEFAULT_NOISE_AMOUNT = 0.03
DEFAULT_NOISE_STRENGTH = 0.2
DEFAULT_DISPLAY_CMAP = "turbo"
DEFAULT_DB_RANGE = 80.0

logging.basicConfig(level=logging.INFO)

def load_font(font_size):
    if DEFAULT_FONT_PATH:
        try:
            return ImageFont.truetype(DEFAULT_FONT_PATH, font_size)
        except Exception as e:
            logging.warning(f"An error occurred while loading the font: {e}")
    return ImageFont.load_default()

def measure_text(text, font, letter_spacing):
    if not text:
        text = " "
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    text_widths = []
    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        text_widths.append(bbox[2] - bbox[0])

    text_width = sum(text_widths) + letter_spacing * max(0, len(text) - 1)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]
    return text_width, text_height, text_widths

def fit_font_size(text, target_height, max_font_size, letter_spacing):
    max_font_size = max(1, max_font_size)
    if DEFAULT_FONT_PATH is None:
        font = ImageFont.load_default()
        text_width, text_height, text_widths = measure_text(text, font, letter_spacing)
        return font, text_width, text_height, text_widths

    low, high = 1, max_font_size
    best = None
    while low <= high:
        mid = (low + high) // 2
        font = load_font(mid)
        text_width, text_height, text_widths = measure_text(text, font, letter_spacing)
        if text_height <= target_height:
            best = (font, text_width, text_height, text_widths)
            low = mid + 1
        else:
            high = mid - 1

    if best is None:
        font = load_font(1)
        text_width, text_height, text_widths = measure_text(text, font, letter_spacing)
        return font, text_width, text_height, text_widths

    font, text_width, text_height, text_widths = best
    return font, text_width, text_height, text_widths

def apply_flutter(image, flutter_px):
    if flutter_px <= 0:
        return image
    height, width = image.shape
    offsets = np.random.randint(-flutter_px, flutter_px + 1, size=width)
    if width >= 9:
        kernel = np.ones(9, dtype=np.float32) / 9.0
        offsets = np.rint(np.convolve(offsets, kernel, mode="same")).astype(int)

    warped = np.zeros_like(image)
    for x in range(width):
        offset = offsets[x]
        if offset == 0:
            warped[:, x] = image[:, x]
        elif offset > 0:
            if offset < height:
                warped[offset:, x] = image[: height - offset, x]
        else:
            shift = -offset
            if shift < height:
                warped[: height - shift, x] = image[shift:, x]
    return warped

def add_text_noise(image, noise_amount, noise_strength):
    if noise_amount <= 0 and noise_strength <= 0:
        return image
    image = image.astype(np.float32)
    mask = image > 0
    if noise_strength > 0:
        noise = np.random.normal(0, noise_strength * 255, image.shape).astype(np.float32)
        image = image + noise * mask
    if noise_amount > 0:
        drop = (np.random.rand(*image.shape) < noise_amount) & mask
        image[drop] = 0
    return np.clip(image, 0, 255).astype(np.uint8)

def spectrogram_to_rgb(S_dB, cmap_name=DEFAULT_DISPLAY_CMAP, db_range=DEFAULT_DB_RANGE):
    if S_dB.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    vmax = np.nanmax(S_dB)
    if not np.isfinite(vmax):
        vmax = 0.0
    vmin = vmax - db_range
    if vmax <= vmin:
        vmin = vmax - 1.0

    norm = (S_dB - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.flipud(norm)
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb

def auto_invert_image(image):
    if image.mean() > 127:
        return 255 - image
    return image

def normalize_image(image):
    image = image.astype(np.float32)
    if image.max() > image.min():
        lo, hi = np.percentile(image, [1, 99])
        if hi > lo:
            image = (image - lo) / (hi - lo)
        else:
            image = image / 255.0
    else:
        image = image / 255.0
    return np.clip(image, 0, 1) * 255

def prepare_image_spectrogram(image, target_height, min_width):
    if image.mode != "L":
        image = image.convert("L")

    width, height = image.size
    if height <= 0:
        raise ValueError("Image height must be greater than 0.")

    scale = target_height / height
    new_width = max(1, int(round(width * scale)))
    image = image.resize((new_width, target_height), Image.LANCZOS)
    image = np.array(image)
    image = auto_invert_image(image)
    image = normalize_image(image).astype(np.uint8)

    if new_width < min_width:
        pad_left = (min_width - new_width) // 2
        pad_right = min_width - new_width - pad_left
        image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
    return image

def text_to_spectrogram_image(
    text,
    base_width=DEFAULT_BASE_WIDTH,
    height=DEFAULT_SPEC_HEIGHT,
    max_font_size=DEFAULT_FONT_SIZE,
    margin=DEFAULT_MARGIN,
    letter_spacing=DEFAULT_LETTER_SPACING,
    flutter_px=DEFAULT_FLUTTER_PX,
    noise_amount=DEFAULT_NOISE_AMOUNT,
    noise_strength=DEFAULT_NOISE_STRENGTH,
):
    safe_margin = max(0, margin + flutter_px)
    target_height = max(1, height - safe_margin * 2)
    font, text_width, text_height, text_widths = fit_font_size(
        text,
        target_height,
        max_font_size,
        letter_spacing,
    )

    width = max(base_width, text_width + safe_margin * 2)
    height = max(height, text_height + safe_margin * 2)

    image = Image.new("L", (width, height), "black")
    draw = ImageDraw.Draw(image)

    text_start_x = (width - text_width) // 2
    text_start_y = (height - text_height) // 2
    min_x = safe_margin
    max_x = max(min_x, width - text_width - safe_margin)
    min_y = safe_margin
    max_y = max(min_y, height - text_height - safe_margin)
    text_start_x = int(np.clip(text_start_x, min_x, max_x))
    text_start_y = int(np.clip(text_start_y, min_y, max_y))

    current_x = text_start_x
    for char, char_width in zip(text, text_widths):
        offset_x = np.random.randint(-flutter_px, flutter_px + 1) if flutter_px > 0 else 0
        offset_y = np.random.randint(-flutter_px, flutter_px + 1) if flutter_px > 0 else 0
        max_x_char = max(min_x, width - char_width - safe_margin)
        char_x = int(np.clip(current_x + offset_x, min_x, max_x_char))
        char_y = int(np.clip(text_start_y + offset_y, min_y, max_y))
        draw.text((char_x, char_y), char, font=font, fill="white")
        current_x += char_width + letter_spacing

    image = np.array(image)
    image = apply_flutter(image, flutter_px)
    image = add_text_noise(image, noise_amount, noise_strength)
    return image

def spectrogram_image_to_audio(image, sr=DEFAULT_SAMPLE_RATE, hop_length=DEFAULT_HOP_LENGTH):
    flipped_image = np.flipud(image)
    S = flipped_image.astype(np.float32) / 255.0
    n_fft = max(2, (S.shape[0] - 1) * 2)
    y = librosa.griffinlim(S, n_iter=64, hop_length=hop_length, win_length=n_fft)
    return y, n_fft

def save_linear_spectrogram(y, sr, n_fft, hop_length):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    rgb = spectrogram_to_rgb(S_dB)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_spectrogram:
        spectrogram_path = temp_spectrogram.name
        Image.fromarray(rgb).save(spectrogram_path)
    return spectrogram_path

def create_audio_with_spectrogram(
    text,
    base_width,
    height,
    max_font_size,
    margin,
    letter_spacing,
    flutter_px,
    noise_amount,
    noise_strength,
):
    spec_image = text_to_spectrogram_image(
        text,
        base_width,
        height,
        max_font_size,
        margin,
        letter_spacing,
        flutter_px,
        noise_amount,
        noise_strength,
    )
    y, n_fft = spectrogram_image_to_audio(spec_image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        sf.write(audio_path, y, DEFAULT_SAMPLE_RATE)

    spectrogram_path = save_linear_spectrogram(y, DEFAULT_SAMPLE_RATE, n_fft, DEFAULT_HOP_LENGTH)

    return audio_path, spectrogram_path

def display_audio_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return save_linear_spectrogram(y, sr, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH)

def create_audio_from_image(image_path, sr=DEFAULT_SAMPLE_RATE):
    if not image_path:
        raise gr.Error("Please upload an image to encode.")

    image = Image.open(image_path)
    spec_image = prepare_image_spectrogram(image, DEFAULT_SPEC_HEIGHT, DEFAULT_BASE_WIDTH)
    y, n_fft = spectrogram_image_to_audio(spec_image, sr)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        img2audio_path = temp_audio.name
        sf.write(img2audio_path, y, sr)

    spectrogram_path = save_linear_spectrogram(y, sr, n_fft, DEFAULT_HOP_LENGTH)
    return img2audio_path, spectrogram_path

def gradio_interface_fn(
    text,
    base_width,
    height,
    max_font_size,
    margin,
    letter_spacing,
    flutter_px,
    noise_amount,
    noise_strength,
):
    audio_path, spectrogram_path = create_audio_with_spectrogram(
        text,
        base_width,
        height,
        max_font_size,
        margin,
        letter_spacing,
        flutter_px,
        noise_amount,
        noise_strength,
    )
    return audio_path, spectrogram_path

def gradio_image_to_audio_fn(upload_image):
    return create_audio_from_image(upload_image)

def gradio_decode_fn(upload_audio):
    return display_audio_spectrogram(upload_audio)

with gr.Blocks(
    title="Audio Steganography",
    css=(
        "footer{display:none !important}"
        "html,body,#root{height:100%}"
        "body{overflow-y:auto !important}"
        "main{overflow:visible !important}"
        ".gradio-container{min-height:100vh;height:auto;overflow:visible;padding-bottom:48px}"
        "#encoded-spectrogram img,#image-encoded-spectrogram img,#decoded-spectrogram img{height:auto !important;max-height:none !important;width:100%;object-fit:contain}"
        "#encoded-spectrogram .image-container,#image-encoded-spectrogram .image-container,#decoded-spectrogram .image-container{height:auto !important;max-height:none !important;overflow:visible !important}"
    ),
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", spacing_size="sm", radius_size="lg"),
) as txt2spec:
    with gr.Tab("Text -> Audio (Encoder)"):
        with gr.Group():
            text = gr.Textbox(lines=2, placeholder="Enter your text:", label="Text", info="Encodes text into a linear spectrogram.")
            with gr.Row(variant="panel"):
                base_width = gr.Slider(minimum=1, maximum=4096, step=1, value=DEFAULT_BASE_WIDTH, label="Image Width", visible=False)
                height = gr.Slider(minimum=1, maximum=4097, step=1, value=DEFAULT_SPEC_HEIGHT, label="Image Height", visible=False)
                max_font_size = gr.Slider(minimum=10, maximum=DEFAULT_FONT_SIZE_CAP, step=1, value=DEFAULT_FONT_SIZE, label="Font size (max)")
                margin = gr.Slider(minimum=0, maximum=200, step=1, value=DEFAULT_MARGIN, label="Indent")
                letter_spacing = gr.Slider(minimum=0, maximum=200, step=1, value=DEFAULT_LETTER_SPACING, label="Letter spacing")
                flutter_px = gr.Slider(minimum=0, maximum=20, step=1, value=DEFAULT_FLUTTER_PX, label="Text flutter (px)")
                noise_amount = gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=DEFAULT_NOISE_AMOUNT, label="Text noise")
                noise_strength = gr.Slider(minimum=0.0, maximum=0.6, step=0.05, value=DEFAULT_NOISE_STRENGTH, label="Noise strength")
            generate_button = gr.Button("Encode text", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            with gr.Group():
                output_audio = gr.Audio(type="filepath", label="Encoded audio")
                output_spectrogram = gr.Image(type="filepath", label="Encoded spectrogram (preview)", elem_id="encoded-spectrogram")

        generate_button.click(
            gradio_interface_fn,
            inputs=[
                text,
                base_width,
                height,
                max_font_size,
                margin,
                letter_spacing,
                flutter_px,
                noise_amount,
                noise_strength,
            ],
            outputs=[output_audio, output_spectrogram],
        )

    with gr.Tab("Image -> Audio (Encoder)"):
        with gr.Group():
            with gr.Column():
                upload_image = gr.Image(type="filepath", label="Upload image (auto-normalize + auto-invert)")
                convert_button = gr.Button("Encode image", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            output_audio_from_image = gr.Audio(type="filepath", label="Encoded audio")
            output_image_spectrogram = gr.Image(type="filepath", label="Encoded spectrogram (preview)", elem_id="image-encoded-spectrogram")

        convert_button.click(
            gradio_image_to_audio_fn,
            inputs=[upload_image],
            outputs=[output_audio_from_image, output_image_spectrogram],
        )

    with gr.Tab("Audio -> Spectrogram (Decoder)"):
        with gr.Group():
            with gr.Column():
                upload_audio = gr.Audio(type="filepath", label="Upload audio", scale=3)
                decode_button = gr.Button("Decode spectrogram", variant="primary", size="lg")

        with gr.Column(variant="panel"):
            decoded_image = gr.Image(type="filepath", label="Decoded spectrogram", elem_id="decoded-spectrogram")

        decode_button.click(gradio_decode_fn, inputs=[upload_audio], outputs=[decoded_image])

txt2spec.launch(share=True)
