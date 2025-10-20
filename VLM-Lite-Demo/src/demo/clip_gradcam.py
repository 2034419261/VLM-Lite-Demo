# src/demo/clip_gradcam.py
import os
import argparse
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from src.models.clip_like import CLIPLikeModel
from src.data.cifar_text import CIFAR10_CLASSES

OUT_DIR = "output/gradcam"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224,224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return img, transform(img_resized).unsqueeze(0)  # PIL.Image, tensor(1,C,H,W)

def find_last_conv(module):
    last_conv = None
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            last_conv = m
            break
    return last_conv

def apply_colormap_on_image(org_img, activation, colormap=cv2.COLORMAP_JET):
    # activation: HxW normalized 0..1
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    org = np.float32(org_img) / 255
    cam = heatmap * 0.5 + org * 0.5
    cam = np.uint8(255 * cam)
    return cam

def gradcam_for_image(model, device, input_tensor, target_text_emb=None, target_class_idx=None):
    """
    input_tensor: (1,3,224,224)
    target_text_emb: (1,D) torch tensor normalized OR target_class_idx as int
    returns: cam (H,W) normalized 0..1
    """
    model.eval()
    # find last conv module
    backbone = model.img_enc.backbone
    last_conv = find_last_conv(backbone)
    if last_conv is None:
        raise RuntimeError("No Conv2d found in backbone for GradCAM")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out
        # register hook to capture grad of this activation during backward
        out.register_hook(lambda grad: gradients.setdefault('value', grad))

    h = last_conv.register_forward_hook(forward_hook)

    # forward pass to get image embedding
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True
    img_emb = model.img_enc(input_tensor)  # (1,D)
    img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-10)

    # get text embedding (target)
    if target_text_emb is None and target_class_idx is None:
        # choose predicted top1 label
        class_ids = torch.arange(len(CIFAR10_CLASSES), device=device, dtype=torch.long)
        txt_embs = model.txt_enc(class_ids)
        txt_embs = txt_embs / (txt_embs.norm(dim=-1, keepdim=True)+1e-10)
        sims = (img_emb @ txt_embs.t())[0]  # (num_classes,)
        target_idx = sims.argmax().item()
        target_text_emb = txt_embs[target_idx].unsqueeze(0)
        print("Auto chosen target class:", CIFAR10_CLASSES[target_idx])
    elif target_class_idx is not None:
        txt = model.txt_enc(torch.tensor([target_class_idx], device=device))
        target_text_emb = txt / (txt.norm(dim=-1, keepdim=True)+1e-10)

    # compute similarity score scalar
    score = (img_emb @ target_text_emb.t()).squeeze()
    # backward to get gradients w.r.t activation
    model.zero_grad()
    if isinstance(score, torch.Tensor):
        score.backward(retain_graph=True)
    else:
        score.backward(retain_graph=True)

    # now get activations and gradients
    if 'value' not in activations or 'value' not in gradients:
        h.remove()
        raise RuntimeError("Failed to capture activations/gradients for GradCAM")
    act = activations['value'].detach().cpu()   # (1,C,H,W)
    grad = gradients['value'].detach().cpu()    # (1,C,H,W)

    # weights: global average pooling of gradients over spatial dims
    weights = grad.mean(dim=(2,3), keepdim=True)  # (1,C,1,1)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1,1,H,W)
    cam = torch.relu(cam)
    cam = cam.squeeze().numpy()
    # normalize to 0..1
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    # remove hook
    h.remove()
    return cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/clip_like_small.pth")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="optional: target text (prefer CIFAR class name)")
    parser.add_argument("--class_idx", type=int, default=None, help="optional: target class index 0-9")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=False).to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("Loaded CKPT:", args.ckpt)
    else:
        print("Warning: checkpoint not found, using randomly initialized model.")

    orig_img, tensor = preprocess_image(args.image_path)
    tensor = tensor.to(device)
    # prepare target embedding if text provided
    target_text_emb = None
    if args.text:
        # map text to CIFAR class id by simple matching
        text = args.text.strip().lower()
        found_idx = None
        for i,c in enumerate(CIFAR10_CLASSES):
            if c == text or c in text or text in c:
                found_idx = i
                break
        if found_idx is None:
            print("Text not matched to known CIFAR classes. Ignoring text.")
        else:
            target_text_emb = model.txt_enc(torch.tensor([found_idx], device=device))
            target_text_emb = target_text_emb / (target_text_emb.norm(dim=-1, keepdim=True)+1e-10)

    cam = gradcam_for_image(model, device, tensor, target_text_emb=target_text_emb, target_class_idx=args.class_idx)
    # resize cam to original image size
    cam_resized = cv2.resize(cam, (orig_img.size[0], orig_img.size[1]))
    overlay = apply_colormap_on_image(np.array(orig_img), cam_resized)
    out_path = args.out if args.out else os.path.join(OUT_DIR, Path(args.image_path).stem + "_gradcam.jpg")
    Image.fromarray(overlay).save(out_path)
    print("Saved GradCAM to:", out_path)

if __name__ == "__main__":
    from pathlib import Path
    main()
