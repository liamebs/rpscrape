"""
inputs_config.py

Defines the schema for model inputs, including:
- Float feature names
- Index-based categorical feature names
- NLP vector size
"""

FLOAT_FEATURES = [
    'age', 'or', 'rpr', 'ts', 'lbs', 'draw',
    'trainer_ovr_win_pct', 'jockey_ovr_win_pct',
    'rpr_rank', 'or_rank', 'rpr_zscore', 'or_zscore',
    'trainer_ovr_profit', 'jockey_ovr_profit',
    'trainer_last_14_win_pct', 'jockey_last_14_win_pct'
]

IDX_FEATURES = [
    'country_idx', 'going_idx', 'sex_idx', 'type_idx',
    'class_label_idx', 'headgear_idx', 'race_class_idx', 'venue_idx'
]

NLP_VECTOR_SIZE = 384  # Adjust based on your embeddings
