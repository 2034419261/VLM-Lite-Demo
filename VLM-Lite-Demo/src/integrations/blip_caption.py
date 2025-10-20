# src/integrations/blip_caption.py
"""
BLIP wrapper using Hugging Face transformers.
Provides:
- load_blip(device, model_name)
- caption_image(pil_image, model, processor, device, gen_kwargs)
- caption_images(list_of_pil_images, ...)
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip(model_name="Salesforce/blip-image-captioning-base", device=None, dtype=torch.float32):
    """
    Load BLIP model + processor from Hugging Face.
    model_name: HF repo id (base or large)
    Returns (model, processor)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor, device

def caption_image(pil_image, model, processor, device, max_length=32, num_beams=3):
    """
    Generate caption for a single PIL.Image (RGB).
    Returns: caption string
    """
    # preprocess
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    # generate
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    # decode - use processor's tokenizer
    caption = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption

def caption_images(pil_images, model, processor, device, **gen_kwargs):
    """
    Batch captions for list of PIL images (naive loop).
    Returns list of strings.
    """
    captions = []
    for img in pil_images:
        cap = caption_image(img, model, processor, device, **gen_kwargs)
        captions.append(cap)
    return captions
