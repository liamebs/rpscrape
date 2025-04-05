# This is a script I'll use to process data so it'll go in '/scripts'.

# Get ChatGPT to create terminal command with 'argparse'.

import pandas as pd
from utils.clean_flattened_df import clean_flattened_dataframe
from utils.add_embedding_indices import add_embedding_indices

def save_model_ready_pickle(input_path, output_path):
    # Load raw flattened .pkl
    df = pd.read_pickle(input_path)
    
    # Clean data types, parse dates, etc.
    df = clean_flattened_dataframe(df)
    
    # Add embedding indices
    df, encoders = add_embedding_indices(df)
    
    # Save enriched model-ready dataframe
    df.to_pickle(output_path)
    print(f"Model-ready DataFrame saved to: {output_path}")
    return output_path

# Example usage:
# save_model_ready_pickle("data/processed/2025-03-29T14-22.pkl", "data/processed/model_ready_flat.pkl")
