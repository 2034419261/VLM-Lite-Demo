# src/models/clip_like.py
import torch
import torch.nn as nn
import timm
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

VOCAB = list(set(" ".join([
  'airplane automobile bird cat deer dog frog horse ship truck'
]).split()))  # characters? we'll do word-level for CIFAR classes
# For CIFAR classes the tokens are single words, so simple mapping:
TEXT_TOKENS = {
    "airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9
}

class SimpleTextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # since each CIFAR class is a single token, just embed class id
        self.embed = nn.Embedding(num_embeddings=10, embedding_dim=embed_dim)
    def forward(self, class_ids):
        # class_ids: (B,) long tensor
        x = self.embed(class_ids)  # (B,embed_dim)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, out_dim=512):
        super().__init__()
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.backbone = m
        feat_dim = m.num_features
        self.proj = nn.Linear(feat_dim, out_dim)
    def forward(self, x):
        f = self.backbone(x)  # (B, feat_dim)
        return self.proj(f)

class CLIPLikeModel(nn.Module):
    def __init__(self, img_model='resnet18', embed_dim=512, pretrained=True):
        super().__init__()
        self.img_enc = ImageEncoder(model_name=img_model, pretrained=pretrained, out_dim=embed_dim)
        self.txt_enc = SimpleTextEncoder(embed_dim=embed_dim)
        # optional temperature param
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    def forward(self, images, class_ids):
        img_emb = self.img_enc(images)   # (B, D)
        txt_emb = self.txt_enc(class_ids) # (B, D)
        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return img_emb, txt_emb, self.logit_scale.exp()
