# src/analysis/visualize_embeddings.py
import argparse, os, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.cifar_text import CIFARTextDataset, CIFAR10_CLASSES
from src.models.clip_like import CLIPLikeModel

def collect_embeddings(model, device):
    ds = CIFARTextDataset(train=False, download=True)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, texts, lbls in dl:
            imgs = imgs.to(device)
            e = model.img_enc(imgs)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu())
            labels.append(lbls)
    return torch.cat(embs, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="output/embeddings_tsne.png")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    embs, labels = collect_embeddings(model, device)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X = tsne.fit_transform(embs)
    plt.figure(figsize=(10,10))
    for cls in range(10):
        idx = labels==cls
        plt.scatter(X[idx,0], X[idx,1], label=CIFAR10_CLASSES[cls], s=5)
    plt.legend()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print("Saved TSNE to", args.out)

if __name__ == "__main__":
    main()
