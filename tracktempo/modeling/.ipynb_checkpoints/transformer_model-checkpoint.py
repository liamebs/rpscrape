import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHeads(nn.Module):
    def __init__(self, label_encoders: dict, float_dim: int, embedding_dim: int, nlp_dim: int):
        super().__init__()

        # Dynamically create an embedding layer for each encoder
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(enc.classes_), embedding_dim) for enc in label_encoders.values()
        ])

        self.proj_float = nn.Linear(float_dim, embedding_dim)
        self.proj_comment = nn.Linear(nlp_dim, embedding_dim)
        self.proj_spotlight = nn.Linear(nlp_dim, embedding_dim)

    def forward(self, float_inputs, idx_inputs, comment_vecs, spotlight_vecs):
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(idx_inputs[..., i]))

        cat_emb = torch.cat(emb_list, dim=-1)
        float_proj = self.proj_float(float_inputs)
        comm_proj = self.proj_comment(comment_vecs)
        spot_proj = self.proj_spotlight(spotlight_vecs)

        return torch.cat([cat_emb, float_proj, comm_proj, spot_proj], dim=-1)


class RaceTransformer(nn.Module):
    def __init__(
        self,
        label_encoders: dict,
        float_dim: int,
        embedding_dim: int,
        nlp_dim: int,
        hidden_dim: int,
        nhead: int,
        num_layers: int
    ):
        super().__init__()

        self.heads = EmbeddingHeads(label_encoders, float_dim, embedding_dim, nlp_dim)

        total_embedding_dim = (embedding_dim * len(label_encoders)) + embedding_dim * 3  # categorical + float + 2 NLP
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_embedding_dim,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(total_embedding_dim, total_embedding_dim),
            nn.ReLU(),
            nn.Linear(total_embedding_dim, 1)  # Output score per horse
        )

    def forward(self, float_feats, idx_feats, comment_vecs, spotlight_vecs, mask):
        x = self.heads(float_feats, idx_feats, comment_vecs, spotlight_vecs)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        return self.output_head(x).squeeze(-1)
