# src/infer_single.py
import argparse, torch
from PIL import Image
from torchvision import transforms
from src.models.clip_like import CLIPLikeModel, TEXT_TOKENS, CIFAR10_CLASSES

def preprocess_image(p):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(p).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPLikeModel(img_model='resnet18', embed_dim=256, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    img = Image.open(args.image_path).convert("RGB")
    x = preprocess_image(img).to(device)
    with torch.no_grad():
        img_emb = model.img_enc(x)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        class_ids = torch.arange(10, device=device)
        txt_emb = model.txt_enc(class_ids)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ txt_emb.t())[0]
        topk = sims.topk(3)
    print("Top predictions:")
    for sc, idx in zip(topk.values.cpu().numpy(), topk.indices.cpu().numpy()):
        print(f"{idx} {CIFAR10_CLASSES[idx]}  score={sc:.4f}")

if __name__ == "__main__":
    main()
