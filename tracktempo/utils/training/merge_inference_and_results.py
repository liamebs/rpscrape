import pandas as pd
from difflib import get_close_matches

def strip_suffix_series(series):
    return (
        series.fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"(\s\([a-z]{2,5}\))+\s*$", "", regex=True)
    )

def merge_inference_and_results(infer_path, results_path, output_path, unmatched_csv_path, missing_races_csv_path, enable_place_flag=False):
    df = pd.read_pickle(infer_path)
    results = pd.read_csv(results_path) if results_path.endswith(".csv") else pd.read_pickle(results_path)

    df['course_clean'] = strip_suffix_series(df['course'])
    df['name_clean'] = strip_suffix_series(df['name'])
    df['race_date'] = pd.to_datetime(df['race_datetime']).dt.date.astype(str)
    df['race_time'] = df['off_time'].astype(str).str.strip()
    df['join_key'] = df['course_clean'] + '|' + df['name_clean'] + '|' + df['race_date'] + '|' + df['race_time']

    results['course_clean'] = strip_suffix_series(results['course'])
    results['horse_clean'] = strip_suffix_series(results['horse'])
    results['date'] = results['date'].astype(str).str.strip()
    results['off'] = results['off'].astype(str).str.strip()
    results['position'] = pd.to_numeric(results['pos'], errors='coerce')
    results['winner_flag'] = (results['pos'] == '1').astype(int)
    if enable_place_flag:
        results['place_flag'] = results['pos'].isin(['1', '2', '3']).astype(int)
    results['join_key'] = results['course_clean'] + '|' + results['horse_clean'] + '|' + results['date'] + '|' + results['off']

    # Exact match
    exact_keys = set(df['join_key']).intersection(set(results['join_key']))
    exact_prerace = df[df['join_key'].isin(exact_keys)]
    exact_postrace = results[results['join_key'].isin(exact_keys)]

    merge_cols = ['join_key', 'position', 'winner_flag']
    if enable_place_flag:
        merge_cols.append('place_flag')

    exact_merged = pd.merge(
        exact_prerace,
        exact_postrace[merge_cols],
        on='join_key',
        how='left'
    )

    # Fuzzy fallback
    df_fuzzy_candidates = df[~df['join_key'].isin(exact_keys)].copy()
    fuzzy_matches = []
    grouped_post = results.groupby(['course_clean', 'date', 'off'])

    for idx, row in df_fuzzy_candidates.iterrows():
        group_key = (row['course_clean'], row['race_date'], row['race_time'])
        if group_key in grouped_post.groups:
            candidates = grouped_post.get_group(group_key)
            matches = get_close_matches(row['name_clean'], candidates['horse_clean'], n=1, cutoff=0.85)
            if matches:
                match = matches[0]
                matched_row = candidates[candidates['horse_clean'] == match].iloc[0]
                merged_row = row.copy()
                merged_row['position'] = matched_row['position']
                merged_row['winner_flag'] = matched_row['winner_flag']
                if enable_place_flag:
                    merged_row['place_flag'] = matched_row['place_flag']
                fuzzy_matches.append(merged_row)

    fuzzy_df = pd.DataFrame(fuzzy_matches)
    final_merged = pd.concat([exact_merged, fuzzy_df], ignore_index=True)

    # Log unmatched prerace
    remaining_unmatched = df[~df['join_key'].isin(final_merged['join_key'])]
    remaining_unmatched[['course', 'name', 'race_date', 'race_time']].drop_duplicates().to_csv(unmatched_csv_path, index=False)

    # Log missing races
    prerace_keys = df[['course_clean', 'race_date', 'race_time']].drop_duplicates()
    postrace_keys = results[['course_clean', 'date', 'off']].drop_duplicates()
    postrace_keys.columns = ['course_clean', 'race_date', 'race_time']

    missing_races = pd.merge(
        prerace_keys,
        postrace_keys,
        on=['course_clean', 'race_date', 'race_time'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    missing_races = pd.merge(
        missing_races,
        df[['course', 'course_clean', 'race_date', 'race_time']].drop_duplicates(),
        on=['course_clean', 'race_date', 'race_time'],
        how='left'
    )[['course', 'race_date', 'race_time']].drop_duplicates()

    missing_races.to_csv(missing_races_csv_path, index=False)

    final_merged.to_pickle(output_path)
    print(f"✅ Final merged dataset saved to {output_path}")
    print(f"📊 Exact matches: {len(exact_merged)} | Fuzzy matches: {len(fuzzy_df)} | Total: {len(df)}")
    print(f"❗ Still unmatched: {len(remaining_unmatched)} saved to {unmatched_csv_path}")
    print(f"❗ Missing race results saved to {missing_races_csv_path}")