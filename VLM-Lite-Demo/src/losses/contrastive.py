# src/losses/contrastive.py
import torch
import torch.nn.functional as F

def clip_contrastive_loss(image_emb, text_emb, logit_scale):
    # image_emb, text_emb: (B, D), already normalized
    logits = torch.matmul(image_emb, text_emb.t()) * logit_scale
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2.0
