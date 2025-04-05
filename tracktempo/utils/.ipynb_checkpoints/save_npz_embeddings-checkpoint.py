
import numpy as np

def save_embeddings_npz(embeddings_dict, output_path):
    """
    Saves a dictionary of embeddings (field → np.ndarray) as a .npz zip archive.

    Parameters:
        embeddings_dict (dict): Keys are field names, values are np.ndarray (e.g., comment, spotlight)
        output_path (str): Path to output .npz file

    Example:
        save_embeddings_npz(embeddings_dict, 'data/processed/text_embeddings.npz')
    """
    np.savez(output_path, **embeddings_dict)
    print(f"[✓] Saved {len(embeddings_dict)} embedding blocks to {output_path}")

def load_embeddings_npz(input_path):
    """
    Loads a .npz archive of embeddings into a dictionary.

    Parameters:
        input_path (str): Path to the .npz file

    Returns:
        dict: field → np.ndarray
    """
    data = np.load(input_path)
    embeddings_dict = {key: data[key] for key in data.files}
    print(f"[✓] Loaded {len(embeddings_dict)} embedding blocks from {input_path}")
    return embeddings_dict
