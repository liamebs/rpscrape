
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def embed_text_fields(df, fields, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Embeds text fields using SentenceTransformer and stores each vector in a single DataFrame column.

    Parameters:
        df (pd.DataFrame): DataFrame with text fields to embed.
        fields (list): List of text field names (e.g., ['comment', 'spotlight']).
        model_name (str): HuggingFace model name to use.
        batch_size (int): Batch size for encoding.

    Returns:
        df (pd.DataFrame): Same DataFrame with new *_vector columns.
        embeddings_dict (dict): field_name -> np.array of embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings_dict = {}

    for field in fields:
        texts = df[field].fillna("").astype(str).tolist()
        print(f"Embedding field: {field}")

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings)
        df[f"{field}_vector"] = list(embeddings)
        embeddings_dict[field] = embeddings

    return df, embeddings_dict

# Example usage:
# df = pd.read_pickle("model_ready_flat.pkl")
# df, embeds = embed_text_fields(df, fields=["comment", "spotlight"])
# df.to_pickle("model_ready_with_text_vectors.pkl")
