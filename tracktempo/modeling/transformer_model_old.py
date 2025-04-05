"""
transformer_model.py

Builds the main model architecture using:
- EmbeddingHeads
- Transformer encoder
- Output head per runner
"""

import torch
import torch.nn as nn
from modeling.embedding_heads import EmbeddingHeads

class RaceTransformer(nn.Module):
    def __init__(self, idx_vocab_sizes, float_dim, nlp_dim, hidden_dim=352, num_heads=4, num_layers=2):
        super().__init__()
        self.heads = EmbeddingHeads(idx_vocab_sizes, float_dim, nlp_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, float_feats, idx_feats, comment_vecs, spotlight_vecs, mask):
        x = self.heads(float_feats, idx_feats, comment_vecs, spotlight_vecs)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        return self.output_head(x).squeeze(-1)
