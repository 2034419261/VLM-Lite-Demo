# src/eval.py
import argparse, os, torch
from torch.utils.data import DataLoader
from src.data.cifar_text import CIFARTextDataset, CIFAR10_CLASSES
from src.models.clip_like import CLIPLikeModel, TEXT_TOKENS

def evaluate_zero_shot(model, device):
    model.eval()
    class_ids = torch.arange(10, device=device, dtype=torch.long)
    with torch.no_grad():
        text_embs = model.txt_enc(class_ids)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    ds = CIFARTextDataset(train=False, download=True)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    correct = total = 0
    for imgs, texts, labels in dl:
        imgs = imgs.to(device)
        with torch.no_grad():
            img_emb = model.img_enc(imgs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = img_emb @ text_embs.t()
            preds = sims.argmax(dim=-1).cpu().numpy()
            correct += (preds == labels.numpy()).sum()
            total += imgs.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    acc = evaluate_zero_shot(model, device)
    print(f"Zero-shot accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
