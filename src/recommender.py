# src/recommender.py
import torch
import numpy as np

def recommend_next_topn(model, idx2item, hist_items, hist_corrs, top_n=5, max_len=50, device="cpu"):
    model.eval()

    items = hist_items[-max_len:]
    corrs = hist_corrs[-max_len:]

    pad_len = max_len - len(items)
    items = [0]*pad_len + items
    corrs = [0]*pad_len + corrs

    items = torch.LongTensor(items).unsqueeze(0).to(device)
    corrs = torch.FloatTensor(corrs).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(items, corrs)
        probs = torch.softmax(logits, dim=1).squeeze()

    topk = torch.topk(probs, top_n)
    raw_probs = topk.values.cpu().numpy()
    indices = topk.indices.cpu().numpy()

    norm_probs = raw_probs / raw_probs.sum()

    rec_items = [idx2item[i] for i in indices]

    return rec_items, norm_probs
