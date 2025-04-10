import pandas as pd
import argparse

def determine_places(num_runners, race_type):
    if num_runners < 5:
        return 0  # Not a valid betting race

    is_handicap = 'handicap' in race_type.lower()
    if 5 <= num_runners <= 7:
        return 2
    elif 8 <= num_runners <= 15:
        return 3 if is_handicap else 3
    elif num_runners >= 16:
        return 4 if is_handicap else 3
    return 0

def evaluate_accuracy(preds_csv, model_csv):
    df_preds = pd.read_csv(preds_csv)
    df_model = pd.read_csv(model_csv)

    # Check if race_type exists
    race_type_cols = ['race_id', 'name', 'position']
    if 'race_type' in df_model.columns:
        race_type_cols.append('race_type')

    df_merged = df_preds.merge(
        df_model[race_type_cols],
        left_on=['race_id', 'runner'],
        right_on=['race_id', 'name'],
        how='left'
    )

    df_merged['is_placed'] = False
    df_merged['is_top1'] = False

    # Infer prediction ranks if not present
    if 'pred_rank' not in df_merged.columns:
        pred_col = None
        for col in ['pred', 'prediction']:
            if col in df_merged.columns:
                pred_col = col
                break

        if pred_col is None:
            raise KeyError("Neither 'pred_rank' nor prediction score column ('pred' or 'prediction') found in dataset.")

        df_merged['pred_rank'] = df_merged.groupby('race_id')[pred_col].rank(ascending=False, method='first')

    for race_id, group in df_merged.groupby('race_id'):
        num_runners = len(group)
        race_type = group['race_type'].iloc[0] if 'race_type' in group.columns and pd.notna(group['race_type'].iloc[0]) else ''
        places = determine_places(num_runners, race_type)

        sorted_preds = group.sort_values('pred_rank')
        top_preds = sorted_preds.head(places)
        top1_pred = sorted_preds.head(1)

        df_merged.loc[top_preds.index, 'is_placed'] = top_preds['position'].isin([1, 2, 3, 4])
        df_merged.loc[top1_pred.index, 'is_top1'] = top1_pred['position'] == 1

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate placed and top-1 accuracy based on UK/IRE racing rules")
    parser.add_argument('--preds', required=True, help="Path to enriched predictions CSV")
    parser.add_argument('--model_data', required=True, help="Path to original training model CSV")
    args = parser.parse_args()

    evaluate_accuracy(args.preds, args.model_data)
