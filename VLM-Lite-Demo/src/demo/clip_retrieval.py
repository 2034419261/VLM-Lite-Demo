# src/demo/clip_retrieval.py
import os
import argparse
import numpy as np
from PIL import Image
from math import ceil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.models.clip_like import CLIPLikeModel
from src.data.cifar_text import CIFAR10_CLASSES, CIFARTextDataset
import torchvision.utils as vutils
import difflib

CACHE_DIR = "output/retrieval"
os.makedirs(CACHE_DIR, exist_ok=True)

def build_image_embeddings(model, device, batch_size=256, cache_path=None):
    """
    计算并缓存 CIFAR10 test image embeddings。
    返回：embeddings (N,D), labels list, PIL images list (resized)
    """
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d['embs'], d['labels'].tolist(), d['imgs'].tolist()
    ds = CIFARTextDataset(train=False, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                         ]))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_embs = []
    all_labels = []
    pil_imgs = []
    model.eval()
    with torch.no_grad():
        for imgs, texts, labels in dl:
            imgs = imgs.to(device)
            emb = model.img_enc(imgs)  # (B,D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
        # need to retrieve original PIL images (untransformed) for visualization
    # load raw PIL test images separately (so visuals are clean)
    raw_ds = datasets.CIFAR10(root="./data", train=False, download=False)
    for i in range(len(raw_ds)):
        img, lbl = raw_ds[i]
        pil_imgs.append(np.array(img.resize((224,224))))
    embs = np.concatenate(all_embs, axis=0)
    if cache_path:
        np.savez_compressed(cache_path, embs=embs, labels=np.array(all_labels), imgs=np.array(pil_imgs))
    return embs, all_labels, pil_imgs

def match_query_to_class(query):
    q = query.strip().lower()
    # direct match
    for i, c in enumerate(CIFAR10_CLASSES):
        if q == c:
            return i
    # substring match
    for i, c in enumerate(CIFAR10_CLASSES):
        if c in q or q in c:
            return i
    # fuzzy match
    matches = difflib.get_close_matches(q, CIFAR10_CLASSES, n=1, cutoff=0.4)
    if matches:
        return CIFAR10_CLASSES.index(matches[0])
    return None

def save_retrieval_grid(top_imgs, scores, out_path):
    """
    top_imgs: list of numpy arrays HWC RGB
    """
    imgs = [Image.fromarray(im) for im in top_imgs]
    # build grid: make them horizontally concatenated
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(out_path)
    print("Saved retrieval result to:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/clip_like_small.pth")
    parser.add_argument("--query", type=str, help="text query (prefer CIFAR class name).")
    parser.add_argument("--image_path", type=str, help="optional: an image to query (image->text).")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cache", type=str, default="output/retrieval/emb_cache.npz")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=False).to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("Loaded CKPT:", args.ckpt)
    else:
        print("Warning: checkpoint not found, using randomly initialized model.")

    # prepare embeddings (cache to speed up)
    embs, labels, pil_imgs = build_image_embeddings(model, device, cache_path=args.cache)
    # normalize (should be already)
    # embs shape (N, D)
    if args.query:
        cls_id = match_query_to_class(args.query)
        if cls_id is None:
            print("Query not matched to CIFAR classes. Try one of:", CIFAR10_CLASSES)
            return
        # compute text embedding of that class via model.txt_enc
        model.eval()
        with torch.no_grad():
            txt_ids = torch.tensor([cls_id], dtype=torch.long, device=device)
            txt_emb = model.txt_enc(txt_ids)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb.cpu().numpy()  # (1,D)
        sims = (embs @ txt_emb.T).squeeze()  # (N,)
        topk_idx = np.argsort(-sims)[:args.topk]
        top_imgs = [pil_imgs[i] for i in topk_idx]
        top_scores = sims[topk_idx]
        base = args.query.replace(" ", "_")
        out_path = os.path.join(CACHE_DIR, f"retrieval_{base}.jpg")
        save_retrieval_grid(top_imgs, top_scores, out_path)
        print("Top results (idx, label, score):")
        for idx, sc in zip(topk_idx, top_scores):
            print(idx, CIFAR10_CLASSES[labels[idx]], float(sc))
    elif args.image_path:
        # image -> class (topk labels)
        img = Image.open(args.image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        x = transform(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            img_emb = model.img_enc(x)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            # compute text embeddings for all class ids
            class_ids = torch.arange(len(CIFAR10_CLASSES), device=device, dtype=torch.long)
            txt_embs = model.txt_enc(class_ids)
            txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
            sims = (img_emb @ txt_embs.t())[0].cpu().numpy()
        topk_idx = np.argsort(-sims)[:args.topk]
        print("Top text matches for given image:")
        for idx in topk_idx:
            print(idx, CIFAR10_CLASSES[idx], float(sims[idx]))
    else:
        print("Provide --query or --image_path. Example:")
        print('python -m src.demo.clip_retrieval --query "cat"')
        print('python -m src.demo.clip_retrieval --image_path samples/cat.png')

if __name__ == "__main__":
    main()
