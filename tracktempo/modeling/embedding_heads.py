"""
embedding_heads.py

Constructs embedding layers for index-based features
and prepares input vectors from float + NLP sources.
"""

import torch
import torch.nn as nn

class EmbeddingHeads(nn.Module):
    def __init__(self, idx_vocab_sizes, float_dim, nlp_dim, emb_dim=32):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in idx_vocab_sizes
        ])
        self.proj_float = nn.Linear(float_dim, emb_dim)
        self.proj_comment = nn.Linear(nlp_dim, emb_dim)
        self.proj_spotlight = nn.Linear(nlp_dim, emb_dim)

    def forward(self, float_inputs, idx_inputs, comment_vecs, spotlight_vecs):
        emb_idx = [emb(idx_inputs[..., i]) for i, emb in enumerate(self.embeddings)]
        cat_emb = torch.cat(emb_idx, dim=-1)
        float_proj = self.proj_float(float_inputs)
        comm_proj = self.proj_comment(comment_vecs)
        spot_proj = self.proj_spotlight(spotlight_vecs)
        return torch.cat([cat_emb, float_proj, comm_proj, spot_proj], dim=-1)
