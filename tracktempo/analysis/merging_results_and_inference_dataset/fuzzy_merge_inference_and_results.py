
import pandas as pd
from rapidfuzz import fuzz, process

from utils.training.merge_inference_and_results import normalize_course, strip_suffix

def fuzzy_match_horses(infer_path, results_path, output_path, threshold=90):
    df = pd.read_csv(infer_path) if infer_path.endswith(".csv") else pd.read_pickle(infer_path)
    results = pd.read_csv(results_path) if results_path.endswith(".csv") else pd.read_pickle(results_path)

    # Normalize
    df["course"] = df["course"].apply(normalize_course)
    df["name"] = strip_suffix(df["name"])
    df["race_date"] = pd.to_datetime(df["race_datetime"]).dt.date.astype(str)
    df["race_time"] = df["off_time"].astype(str).str.strip()

    results["course"] = results["course"].apply(normalize_course)
    results["horse"] = strip_suffix(results["horse"])
    results["date"] = results["date"].astype(str).str.strip()
    results["off"] = results["off"].astype(str).str.strip()
    results["position"] = pd.to_numeric(results["pos"], errors="coerce")
    results["winner_flag"] = (results["pos"] == "1").astype(int)

    merged_rows = []

    # Merge based on shared race
    race_keys = ["course", "race_date", "race_time"]
    df_grouped = df.groupby(race_keys)
    results_grouped = results.groupby(["course", "date", "off"])

    for key, df_race in df_grouped:
        course, race_date, race_time = key
        key_res = (course, race_date, race_time)
        if key_res not in results_grouped.groups:
            continue
        res_race = results_grouped.get_group(key_res)

        infer_names = df_race["name"].tolist()
        result_names = res_race["horse"].tolist()

        # Fuzzy match within this race
        matches = {}
        for name in infer_names:
            match, score, idx = process.extractOne(name, result_names, scorer=fuzz.ratio)
            if score >= threshold:
                matches[name] = (match, score)

        for i, row in df_race.iterrows():
            name = row["name"]
            if name in matches:
                matched_name = matches[name][0]
                match_row = res_race[res_race["horse"] == matched_name].iloc[0]
                for col in ["position", "winner_flag"]:
                    row[col] = match_row[col]
                merged_rows.append(row)

    result_df = pd.DataFrame(merged_rows)
    result_df.to_pickle(output_path)
    print(f"✅ Fuzzy-matched dataset saved to {output_path}")
    print(f"📊 Matched shape: {result_df.shape}")
    print(f"📈 Match coverage: {round(len(result_df)/len(df)*100, 2)}%")
