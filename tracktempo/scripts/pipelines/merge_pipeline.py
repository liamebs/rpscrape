"""
TrackTempo Inference-to-Training Merge Pipeline
Appends race results to the inference dataset for model training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
from utils.training.merge_inference_and_results import merge_inference_and_results

def main():
    parser = argparse.ArgumentParser(description="Merge inference dataset with postrace results")
    parser.add_argument("--infer", required=True, help="Path to prerace inference .pkl file")
    parser.add_argument("--results", required=True, help="Path to scraped postrace .csv results file")
    parser.add_argument("--output", required=True, help="Path to save merged .pkl")
    parser.add_argument("--unmatched_csv", default="data/processed/unmatched.csv", help="Path to log unmatched horses")
    parser.add_argument("--missing_races_csv", default="data/processed/missing_races.csv", help="Path to log missing races")
    parser.add_argument("--enable_place_flag", action="store_true", help="Enable place logic in merge")
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
    if args.output.endswith(".pkl"):
        output_base = args.output.replace(".pkl", "")
    else:
        output_base = args.output
    final_output_path = f"{output_base}_{timestamp}.pkl"


    print("🔀 Starting merge of inference + results...")
    merge_inference_and_results(
        infer_path=args.infer,
        results_path=args.results,
        output_path=final_output_path,
        unmatched_csv_path=args.unmatched_csv,
        missing_races_csv_path=args.missing_races_csv,
        enable_place_flag=args.enable_place_flag
    )
    print("✅ Merge complete!")

if __name__ == "__main__":
    main()
