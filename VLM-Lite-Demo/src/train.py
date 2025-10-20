# src/train.py
import torch, random, numpy as np
from torch.utils.data import DataLoader
from src.data.cifar_text import CIFARTextDataset, CIFAR10_CLASSES
from src.models.clip_like import CLIPLikeModel, TEXT_TOKENS
from src.losses.contrastive import clip_contrastive_loss
import torch.optim as optim
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def text_to_ids(texts):
    # texts: list of class name strings
    ids = [TEXT_TOKENS[t] for t in texts]
    return torch.tensor(ids, dtype=torch.long)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, texts, labels in loader:
        imgs = imgs.to(device)
        label_ids = torch.tensor([TEXT_TOKENS[t] for t in texts], dtype=torch.long, device=device)
        optimizer.zero_grad()
        img_emb, txt_emb, logit_scale = model(imgs, label_ids)
        loss = clip_contrastive_loss(img_emb, txt_emb, logit_scale)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_zero_shot(model, device):
    # Build text embeddings for all class names (zero-shot)
    model.eval()
    with torch.no_grad():
        class_ids = torch.arange(10, device=device, dtype=torch.long)
        # create dummy images? We'll just compute text embeddings and test on test set
        text_embs = model.txt_enc(class_ids)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    # Evaluate on test set: compute image embeddings and nearest neighbor
    ds = CIFARTextDataset(train=False, download=True)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    correct = total = 0
    for imgs, texts, labels in dl:
        imgs = imgs.to(device)
        with torch.no_grad():
            img_emb = model.img_enc(imgs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = img_emb @ text_embs.t()  # (B,10)
            preds = sims.argmax(dim=-1).cpu().numpy()
            correct += (preds == labels.numpy()).sum()
            total += imgs.size(0)
    return correct / total

def main():
    seed_everything(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CIFARTextDataset(train=True, download=True)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 5  # small quick demo
    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        acc = evaluate_zero_shot(model, device)
        print(f"Epoch {epoch}: loss={loss:.4f}, zero-shot acc={acc:.4f}")
    # save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/clip_like_small.pth')

if __name__ == '__main__':
    main()
