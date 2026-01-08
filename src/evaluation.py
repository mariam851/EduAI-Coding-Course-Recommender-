# src/evaluation.py
import torch

def recall_at_k(model, loader, k=5, device="cpu"):
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for items, corrects, y_true in loader:
            items, corrects, y_true = items.to(device), corrects.to(device), y_true.to(device)
            logits = model(items, corrects)
            topk = torch.topk(logits, k, dim=1).indices
            for i in range(len(y_true)):
                hits += int(y_true[i] in topk[i])
                total += 1
    return hits / total if total > 0 else 0
