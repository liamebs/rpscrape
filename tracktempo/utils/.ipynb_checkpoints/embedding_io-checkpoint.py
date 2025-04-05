
import numpy as np

def save_embeddings_npz(embeddings_dict, output_path):
    """
    Saves a dictionary of embeddings (field → np.ndarray) as a .npz zip archive.
    """
    np.savez(output_path, **embeddings_dict)
    print(f"[✓] Saved {len(embeddings_dict)} embedding blocks to {output_path}")

def load_embeddings_npz(input_path, expected_schema=None):
    """
    Loads a .npz archive of embeddings into a dictionary with optional schema validation.

    Parameters:
        input_path (str): Path to the .npz file
        expected_schema (dict): Optional dict {field_name: expected_shape}

    Returns:
        dict: field → np.ndarray
    """
    data = np.load(input_path)
    embeddings_dict = {key: data[key] for key in data.files}
    print(f"[✓] Loaded {len(embeddings_dict)} embedding blocks from {input_path}")

    if expected_schema:
        for field, expected_shape in expected_schema.items():
            if field not in embeddings_dict:
                raise ValueError(f"[!] Missing embedding field: '{field}'")
            if embeddings_dict[field].shape != expected_shape:
                raise ValueError(
                    f"[!] Shape mismatch for '{field}': expected {expected_shape}, got {embeddings_dict[field].shape}"
                )
        print("[✓] Schema validation passed.")

    return embeddings_dict
