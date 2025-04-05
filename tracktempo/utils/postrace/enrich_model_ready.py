"""
enrich_model_ready.py

Utility for enriching the model-ready DataFrame with post-race result data.

Functions:
- enrich_model_ready_with_postrace: Aligns scraped model-ready rows with parsed result outcomes.

Expected Input:
- model_df: Flattened, cleaned race data (one row per horse)
- post_df: Cleaned postrace output from `parse_postrace_fields`

Output:
- Augmented model_df with:
    - 'position', 'win_flag', 'rpr_postrace', 'decimal_odds_postrace', 'non_runner_flag'

Example:
from utils.postrace.enrich_model_ready import enrich_model_ready_with_postrace
model_df = enrich_model_ready_with_postrace(model_df, postrace_df)

"""

import pandas as pd

def enrich_model_ready_with_postrace(model_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.copy()
    df["horse_clean"] = df["name"].str.lower().str.strip()
    df["course_clean"] = df["course"].str.lower().str.replace(" ", "").str.strip()

    df = df.merge(
        post_df[
            [
                "race_date",
                "off_time",
                "course_clean",
                "horse_clean",
                "position",
                "rpr_postrace",
                "decimal_odds",
                "non_runner_flag",
            ]
        ],
        on=["race_date", "off_time", "course_clean", "horse_clean"],
        how="left",
    )

    df = df.rename(columns={"decimal_odds": "decimal_odds_postrace"})
    df["win_flag"] = (df["position"] == 1).astype(int)
    return df
