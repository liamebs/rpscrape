"""
parse_postrace_csv.py

Utility for parsing raw post-race result data.

Functions:
- parse_postrace_fields: Cleans and extracts structured info from post-race result CSVs.

Expected Input:
- DataFrame with raw post-race fields, possibly scraped from website or feed

Output:
- Cleaned DataFrame with columns:
    - 'horse_clean', 'decimal_odds', 'position', 'non_runner_flag', 'rpr_postrace'
    - 'race_date', 'off_time', 'course_clean'

Example:
from utils.postrace.parse_postrace_csv import parse_postrace_fields
df_clean = parse_postrace_fields(df_raw)

"""

import pandas as pd

def parse_postrace_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["horse_clean"] = df["horse"].str.lower().str.strip()
    df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["non_runner_flag"] = df["position"].isna().astype(bool)
    df["rpr_postrace"] = pd.to_numeric(df["rpr"], errors="coerce")

    df["race_date"] = pd.to_datetime(df["race_date"]).dt.date
    df["off_time"] = pd.to_datetime(df["off_time"], errors="coerce").dt.time
    df["course_clean"] = df["course"].str.lower().str.replace(" ", "").str.strip()

    return df
