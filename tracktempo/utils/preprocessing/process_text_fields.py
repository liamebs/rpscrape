
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re

def extract_race_phrases(df, text_fields=["comment", "spotlight"]):
    feature_defs = {
        "mentions_course_win": r"course winner",
        "mentions_trip_change": r"up in trip|down in trip",
        "mentions_surface": r"back to turf|switched to a[.-]?w",
        "mentions_front_runner": r"make all|lead from start|set the pace",
        "mentions_layoff": r"last seen \d+ days ago",
        "mentions_first_time_headgear": r"first time (?:blinkers|visor|cheekpieces|hood|tongue[- ]tie)",
        "mentions_class_drop": r"(?:down in class|easier race|dropped in class)",
        "mentions_class_rise": r"(?:up in class|tougher task|raised in class)",
        "mentions_course_form": r"(?:won here before|ran well at|previous course win)",
        "mentions_distance_form": r"(?:won over trip|distance winner|effective at \d+f)",
        "mentions_ground_form": r"(?:won on|handles|prefers) (?:soft|firm|good)",
        "mentions_classy_rival": r"(?:beaten by (?:Group|Listed)|tougher opposition|strong formline)",
        "mentions_fitness_query": r"(?:may need (?:the )?run|returns? from (?:a )?(?:break|layoff)|first run in \d+ days)",
        "mentions_positive_trainer_note": r"(?:trainer|yard) in form|good strike rate|flying",
        "mentions_jockey_combo": r"(?:jockey booking|rides again|partnered before)",
        "mentions_improver_flag": r"(?:progressive|on the up|still improving)",
        "mentions_loser_flag": r"(?:hard to win with|often placed|long losing run)"
    }

    added_features = []
    for feature_name, pattern in feature_defs.items():
        for field in text_fields:
            col_name = f"{feature_name}_{field}"
            df[col_name] = df[field].str.contains(pattern, flags=re.IGNORECASE, na=False).astype(int)
            added_features.append(col_name)

    return df, added_features

def process_text_fields(df, fields, model_name='all-MiniLM-L6-v2', batch_size=32, enable_regex=True):
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

    regex_features = []
    if enable_regex:
        df, regex_features = extract_race_phrases(df, text_fields=fields)

    return df, embeddings_dict, regex_features
