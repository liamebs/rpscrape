
import pandas as pd
import re

def extract_race_phrases(df, text_fields=["comment", "spotlight"]):
    """
    Adds binary flags to the DataFrame based on common race-related phrases.

    Parameters:
        df (pd.DataFrame): DataFrame with comment/spotlight fields.
        text_fields (list): Text columns to scan for phrases.

    Returns:
        df (pd.DataFrame): Updated DataFrame with new binary feature columns.
        list: List of added feature column names.
    """
    feature_defs = {
        # Original 8
        "mentions_course_win": r"course winner",
        "mentions_trip_change": r"up in trip|down in trip",
        "mentions_surface": r"back to turf|switched to a[.-]?w",
        "mentions_front_runner": r"make all|lead from start|set the pace",
        "mentions_layoff": r"last seen \d+ days ago",
        "mentions_first_time_headgear": r"first time (blinkers|visor|cheekpieces|hood|tongue[- ]tie)",
        "mentions_class_drop": r"(down in class|easier race|dropped in class)",
        "mentions_class_rise": r"(up in class|tougher task|raised in class)",

        # Expanded features
        "mentions_course_form": r"(won here before|ran well at|previous course win)",
        "mentions_distance_form": r"(won over trip|distance winner|effective at \d+f)",
        "mentions_ground_form": r"(won on|handles|prefers) (soft|firm|good)",
        "mentions_classy_rival": r"(beaten by (Group|Listed)|tougher opposition|strong formline)",
        "mentions_fitness_query": r"(may need (the )?run|returns? from (a )?(break|layoff)|first run in \d+ days)",
        "mentions_positive_trainer_note": r"(trainer|yard) in form|good strike rate|flying",
        "mentions_jockey_combo": r"(jockey booking|rides again|partnered before)",
        "mentions_improver_flag": r"(progressive|on the up|still improving)",
        "mentions_loser_flag": r"(hard to win with|often placed|long losing run)"
    }

    added_features = []

    for feature_name, pattern in feature_defs.items():
        for field in text_fields:
            col_name = f"{feature_name}_{field}"
            df[col_name] = df[field].str.contains(pattern, flags=re.IGNORECASE, na=False).astype(int)
            added_features.append(col_name)

    return df, added_features

# Example usage:
# df = pd.read_pickle("model_ready_flat.pkl")
# df, new_features = extract_race_phrases(df, text_fields=["comment", "spotlight"])
