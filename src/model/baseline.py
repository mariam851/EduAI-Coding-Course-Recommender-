# src/model/baseline.py
from collections import Counter

def most_frequent_baseline(df, top_k=10):
    all_items = df["item_id"].tolist()
    counter = Counter(all_items)
    top_items = [item for item, _ in counter.most_common(top_k)]
    return top_items
