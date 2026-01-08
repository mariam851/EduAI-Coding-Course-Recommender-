# src/feature_engineering.py
import numpy as np

def build_sequences(df, max_len=50):
    """Build history sequences of items + correctness"""
    item2idx = {item: idx+1 for idx, item in enumerate(df["item_id"].unique())}
    idx2item = {v: k for k, v in item2idx.items()}
    
    samples = []
    for _, g in df.groupby("subject_id"):
        items = g["item_id"].map(item2idx).tolist()
        corrects = g["is_correct"].astype(int).tolist()
        for i in range(1, len(items)):
            hist_items = items[:i][-max_len:]
            hist_corrs = corrects[:i][-max_len:]
            pad_len = max_len - len(hist_items)
            hist_items = [0]*pad_len + hist_items
            hist_corrs = [0]*pad_len + hist_corrs
            samples.append((hist_items, hist_corrs, items[i]))
    
    X_items = np.array([s[0] for s in samples])
    X_corrs = np.array([s[1] for s in samples])
    y = np.array([s[2] for s in samples])
    
    return X_items, X_corrs, y, item2idx, idx2item
