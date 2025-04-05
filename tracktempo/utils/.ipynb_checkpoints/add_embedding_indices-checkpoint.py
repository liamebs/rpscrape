
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def add_embedding_indices(df, columns=None):
    """
    Add embedding-ready integer indices for categorical columns.
    Returns updated DataFrame and a dictionary of fitted LabelEncoders.
    """
    if columns is None:
        # Default columns to embed if not provided
        columns = [
            "country", "going", "sex", "type", "class_label",
            "headgear", "race_class", "venue"
        ]
    
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_idx"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders
