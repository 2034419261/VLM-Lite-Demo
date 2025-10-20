# src/demo/create_caption_video.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import cv2
import torch
from torchvision import transforms, datasets
from src.integrations.blip_caption import load_blip, caption_images
from tqdm import tqdm


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUTPUT_DIR, "blip_demo.mp4")

def load_cifar_images(n=16):
    # load first n images from CIFAR10 test
    ds = datasets.CIFAR10(root="./data", train=False, download=True)
    imgs = []
    for i in range(min(n, len(ds))):
        arr, label = ds[i]
        # arr is PIL.Image (or numpy?) - CIFAR returns PIL.Image by default if no transform
        if not isinstance(arr, Image.Image):
            arr = Image.fromarray(arr)
        imgs.append(arr.convert("RGB"))
    return imgs

def overlay_text_on_image(pil_img, text, font=None, rect_height=60):
    """
    Draw semi-transparent rectangle at bottom and put text.
    Returns PIL.Image RGB
    Robust to different Pillow versions (uses textbbox/textsize/font.getsize fallback).
    """
    img = pil_img.copy().convert("RGBA")
    w, h = img.size
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # rectangle background (semi-transparent black)
    rect_y0 = h - rect_height
    draw.rectangle([(0, rect_y0), (w, h)], fill=(0, 0, 0, 150))

    # font fallback
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        except Exception:
            font = ImageFont.load_default()

    # measure text size robustly
    text_w = text_h = None
    try:
        # Pillow >= 8.0: textbbox gives precise bbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        try:
            # older Pillow: textsize may exist
            text_w, text_h = draw.textsize(text, font=font)
        except Exception:
            try:
                # fallback to font methods
                text_w, text_h = font.getsize(text)
            except Exception:
                # last resort: estimate
                text_w = min(w - 10, len(text) * 8)
                text_h = 20

    # position: center horizontally, vertically center inside rect
    x = max(5, (w - text_w) // 2)
    y = rect_y0 + max(5, (rect_height - text_h) // 2)
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out

def images_to_video(frames_pil, video_path, fps=1):
    """
    frames_pil: list of PIL.Image (RGB)
    Save to video_path using OpenCV VideoWriter
    """
    # convert first frame to numpy to get size
    h, w = frames_pil[0].size[1], frames_pil[0].size[0]
    # OpenCV expects (w,h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w,h))
    for img in frames_pil:
        arr = np.array(img)  # HWC RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(f"Saved video to {video_path}")

def main(num_images=16, fps=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading BLIP model (this will download weights if not cached)...")
    model, processor, device = load_blip(device=device)
    print("Loading images from CIFAR10 test set...")
    imgs = load_cifar_images(n=num_images)
    print(f"Got {len(imgs)} images. Generating captions...")
    # BLIP expects PIL images; we will pass them directly
    captions = caption_images(imgs, model, processor, device, max_length=30, num_beams=3)
    print("Composed captions. Rendering frames...")
    frames = []
    for pil_img, cap in zip(imgs, captions):
        # resize image to processor expected size to make nicer video frames:
        target_size = (384, 384)
        img_resized = pil_img.resize(target_size, Image.BICUBIC)
        frame = overlay_text_on_image(img_resized, cap)
        frames.append(frame)
    print("Writing video...")
    images_to_video(frames, VIDEO_PATH, fps=fps)
    print("Done.")

if __name__ == "__main__":
    main(num_images=16, fps=1)
