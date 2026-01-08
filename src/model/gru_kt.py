# src/model/gru_kt.py
import torch
import torch.nn as nn

class GRUKT(nn.Module):
    def __init__(self, n_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + 1, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, n_items)

    def forward(self, items, corrects):
        emb = self.item_emb(items)
        corrects = corrects.unsqueeze(-1)
        x = torch.cat([emb, corrects], dim=-1)
        _, h = self.gru(x)
        h = self.dropout(h.squeeze(0))
        return self.fc(h)
