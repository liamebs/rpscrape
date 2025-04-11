import pandas as pd
import argparse
from pathlib import Path

def determine_places(num_runners, race_indicator):
    """
    Determine the number of places that count as 'placed' based on the number of runners
    and whether the race is a handicap. The handicap indicator is provided in the race_indicator
    parameter, which can come from 'race_type' or 'class_label'.
    """
    if num_runners < 5:
        return 0  # Not a valid betting race

    # Check if race_indicator indicates a handicap race (case-insensitive)
    is_handicap = 'handicap' in race_indicator.lower() if race_indicator else False
    if 5 <= num_runners <= 7:
        return 2
    elif 8 <= num_runners <= 15:
        return 3  # Handicap or not, rules assume 3 places
    elif num_runners >= 16:
        return 4 if is_handicap else 3
    return 0

def evaluate_accuracy(preds_csv, model_csv):
    # Load predictions and model (race results) data
    df_preds = pd.read_csv(preds_csv)
    df_model = pd.read_csv(model_csv)

    # Define the required columns from the model CSV.
    # We expect race_id, name (runner name), and position.
    # Optionally, columns such as race_type, class_label, date, course, and race_time.
    race_cols = ['race_id', 'name', 'position']
    for col in ['race_type', 'class_label', 'date', 'course', 'race_time']:
        if col in df_model.columns:
            race_cols.append(col)

    # Merge predictions with the model data on race_id and the runner's name.
    df_merged = df_preds.merge(
        df_model[race_cols],
        left_on=['race_id', 'runner'],   # 'runner' is standardized in your predictions CSV
        right_on=['race_id', 'name'],
        how='left'
    )

    # Initialize flags for evaluation
    df_merged['is_placed'] = False
    df_merged['is_top1'] = False

    # Generate prediction ranks if not already provided.
    if 'pred_rank' not in df_merged.columns:
        pred_col = None
        # Look for common prediction score column names.
        for col in ['pred', 'prediction']:
            if col in df_merged.columns:
                pred_col = col
                break
        # Fallback to using the 'model_score' column from your enriched predictions
        if pred_col is None and 'model_score' in df_merged.columns:
            pred_col = 'model_score'
        if pred_col is None:
            raise KeyError("Neither 'pred_rank' nor a prediction score column ('pred', 'prediction', or 'model_score') found in dataset.")
        
        # Rank predictions within each race (highest score gets rank 1)
        df_merged['pred_rank'] = df_merged.groupby('race_id')[pred_col].rank(ascending=False, method='first')

    # Evaluate each race individually
    for race_id, group in df_merged.groupby('race_id'):
        num_runners = len(group)
        # Determine the handicap indicator.
        # Prefer 'race_type' if it exists, otherwise use 'class_label'.
        if 'race_type' in group.columns and pd.notna(group['race_type'].iloc[0]):
            race_indicator = group['race_type'].iloc[0]
        elif 'class_label' in group.columns and pd.notna(group['class_label'].iloc[0]):
            race_indicator = group['class_label'].iloc[0]
        else:
            race_indicator = ''
        places = determine_places(num_runners, race_indicator)
        
        # Sort runners in this race by their predicted rank
        sorted_preds = group.sort_values('pred_rank')
        # Get the top predicted runners corresponding to the awarded places
        top_preds = sorted_preds.head(places)
        top1_pred = sorted_preds.head(1)
        
        # Determine valid finishing positions (1-based) that count as 'placed'
        valid_positions = list(range(1, places + 1))
        # Flag the runners as 'placed' if their actual finishing position is within the valid positions
        df_merged.loc[top_preds.index, 'is_placed'] = top_preds['position'].isin(valid_positions)
        # Flag the top-ranked runner as top1 if its actual finishing position is 1
        df_merged.loc[top1_pred.index, 'is_top1'] = top1_pred['position'] == 1

    # Compute overall accuracy metrics
    total_races = df_merged['race_id'].nunique()
    correct_placed = df_merged.groupby('race_id')['is_placed'].any().sum()
    correct_top1 = df_merged.groupby('race_id')['is_top1'].any().sum()
    
    placed_accuracy = correct_placed / total_races
    top1_accuracy = correct_top1 / total_races

    print(f"Total races: {total_races}")
    print(f"Correctly placed predictions: {correct_placed}")
    print(f"Placed accuracy: {placed_accuracy:.2%}")
    print(f"Correct top-1 predictions: {correct_top1}")
    print(f"Top-1 accuracy: {top1_accuracy:.2%}")

    # --- Generate side-by-side comparison CSV ---
    # Build list of columns in a logical order.
    # Include race-level information first (if available), then runner details and evaluation metrics.
    comparison_cols = ['race_id']
    for col in ['date', 'course', 'race_time', 'race_type', 'class_label']:
        if col in df_merged.columns:
            comparison_cols.append(col)
    comparison_cols.extend(['runner', 'model_score', 'pred_rank', 'position', 'is_placed', 'is_top1'])
    
    # Sort the merged DataFrame by race_id and predicted rank
    df_comparison = df_merged[comparison_cols].sort_values(['race_id', 'pred_rank'])
    
    # Recommend outputting to a dedicated analysis folder outside of your raw utils folder.
    output_folder = Path("analysis")
    output_folder.mkdir(parents=True, exist_ok=True)
    comparison_output = output_folder / "side_by_side_comparison.csv"
    df_comparison.to_csv(comparison_output, index=False)
    print(f"Side-by-side comparison saved to {comparison_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate placed and top-1 accuracy based on UK/IRE racing rules")
    parser.add_argument('--preds', required=True, help="Path to enriched predictions CSV")
    parser.add_argument('--model_data', required=True, help="Path to original training model CSV")
    args = parser.parse_args()
    
    evaluate_accuracy(args.preds, args.model_data)
