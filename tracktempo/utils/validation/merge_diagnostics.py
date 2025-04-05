"""
utils/validation/merge_diagnostics.py

Tools to assess quality of model_ready_train after merging results.
"""

import pandas as pd

def analyze_merge_quality(df):
    total = len(df)
    matched = df["position"].notna().sum()
    missing = df["position"].isna().sum()
    percent_matched = round(matched / total * 100, 2)

    print(f"📊 Total rows: {total}")
    print(f"✅ Matched rows (with position): {matched}")
    print(f"❌ Missing position rows: {missing}")
    print(f"📈 Merge coverage: {percent_matched}%")

def summarize_missing_cases(df, max_rows=10):
    missing = df[df["position"].isna()]
    if missing.empty:
        print("🎉 No missing position cases.")
        return

    summary = (
        missing[["course", "name", "race_date", "race_time"]]
        .value_counts()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .head(max_rows)
    )
    print("\n🔍 Top missing entries:")
    print(summary.to_string(index=False))

def summarize_dnf_labels(results_df, pos_col="pos"):
    """
    Optional: Analyze scraped results to see how many entries have DNF labels.
    """
    if pos_col not in results_df.columns:
        print(f"⚠️ Column '{pos_col}' not in results.")
        return

    dnf_labels = results_df[pos_col].value_counts().loc[lambda x: ~x.index.str.fullmatch(r"\d+")]
    if dnf_labels.empty:
        print("🎯 No DNF-style labels detected.")
    else:
        print("\n🏥 DNF-style position labels:")
        print(dnf_labels)
