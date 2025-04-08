import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from .flatten_day import inject_country_and_flatten, add_relative_features

# ANSI color codes for styled terminal output
class LogColors:
    INFO = "\033[94m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    SUCCESS = "\033[92m"
    END = "\033[0m"

def log(msg, level="info"):
    color = {
        "info": LogColors.INFO,
        "warning": LogColors.WARNING,
        "error": LogColors.ERROR,
        "success": LogColors.SUCCESS
    }.get(level, "")
    print(f"{color}{msg}{LogColors.END}")

def run_batch_flatten():
    # Directory paths
    raw_dir = Path("./data/raw/")
    output_dir = Path("./data/processed/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage
    all_flattened = []
    file_count = 0
    total_runners = 0

    # Process each JSON file
    for json_file in raw_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            log(f"[ERROR] Failed to load JSON: {json_file.name}", level="error")
            continue

        try:
            df = inject_country_and_flatten(data)
        except Exception as e:
            log(f"[ERROR] Flattening failed for {json_file.name}: {str(e)}", level="error")
            continue

        if df.empty:
            log(f"[WARNING] No runners found in {json_file.name}", level="warning")
            continue

        # Add non_runner_flag based on pre-race field (safe fallback)
        if "runner_state" in df.columns:
            df["non_runner_flag"] = df["runner_state"].astype(str).str.lower().isin(["nr", "wd", "nonrunner"])
        else:
            df["non_runner_flag"] = False

        runner_count = len(df)
        total_runners += runner_count
        file_count += 1
        log(f"[INFO] {json_file.name}: {runner_count} runners processed", level="info")

        valid_horse_ids = df['horse_id_valid'].sum()
        class_num_present = df['class_num'].notnull().sum()
        log(f"        Valid horse IDs: {valid_horse_ids}/{runner_count}", level="info")
        log(f"        class_num present: {class_num_present}/{runner_count}", level="info")

        all_flattened.append(df)

    # Concatenate and save
    if all_flattened:
        final_df = pd.concat(all_flattened, ignore_index=True)
        final_df = add_relative_features(final_df)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
        csv_file = output_dir / f"flattened_json_{timestamp}.csv"
        pkl_file = output_dir / f"flattened_json_{timestamp}.pkl"

        final_df.to_csv(csv_file, index=False)
        final_df.to_pickle(pkl_file)

        log(f"[SUCCESS] Flattened {total_runners} runners from {file_count} files", level="success")
        log(f"[✓] CSV saved to {csv_file}", level="success")
        log(f"[✓] Pickle saved to {pkl_file}", level="success")
    else:
        log("[ERROR] No valid runners processed. No file was saved.", level="error")

if __name__ == "__main__":
    run_batch_flatten()
