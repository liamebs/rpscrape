import json
import re
import pandas as pd
from pathlib import Path

def parse_race_class(raw):
    result = {
        'race_class': raw,
        'class_num': None,
        'class_label': None
    }
    if not raw:
        return result

    match = re.search(r'Class (\d)', raw)
    if match:
        result['class_num'] = int(match.group(1))

    if 'Group' in raw:
        result['class_label'] = 'Group'
    elif 'Listed' in raw:
        result['class_label'] = 'Listed'
    elif 'Maiden' in raw:
        result['class_label'] = 'Maiden'
    elif 'Novice' in raw:
        result['class_label'] = 'Novice'
    elif 'Nursery' in raw:
        result['class_label'] = 'Nursery'
    else:
        result['class_label'] = 'Handicap'

    return result

def add_presence_flags(features, field_list):
    for field in field_list:
        val = features.get(field)
        features[f'has_{field}'] = int(bool(val and str(val).strip()))
    return features

def add_relative_features(df):
    relative_fields = [
        {'col': 'rpr', 'rank_col': 'rpr_rank', 'z_col': 'rpr_zscore'},
        {'col': 'or', 'rank_col': 'or_rank', 'z_col': 'or_zscore'}
    ]

    for field in relative_fields:
        col, rank_col, z_col = field['col'], field['rank_col'], field['z_col']
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[rank_col] = df.groupby('race_id')[col].rank(method='min', ascending=False)
        df[z_col] = df.groupby('race_id')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return df

def clean_percent(val):
    if isinstance(val, str) and val.endswith('%'):
        try:
            return float(val.strip('%')) / 100
        except ValueError:
            return None
    return None

def clean_float(val):
    try:
        return float(str(val).replace('+', '').strip())
    except (ValueError, AttributeError):
        return None

def extract_features_v2(runner_json, race_context):
    features = {}

    raw_id = runner_json.get('horse_id')
    features['horse_id'] = str(raw_id).strip()
    features['horse_id_valid'] = int(bool(re.match(r'^\d{7}$', features['horse_id'])))

    features['course'] = race_context.get('course')
    features['country'] = race_context.get('country')
    features['going'] = race_context.get('going')
    features['GoingStick'] = race_context.get('GoingStick')
    features['distance_f'] = race_context.get('distance_f')
    features['field_size'] = race_context.get('field_size')
    features['race_class'] = race_context.get('race_class')
    features['class_num'] = race_context.get('class_num')
    features['class_label'] = race_context.get('class_label')
    features['type'] = race_context.get('type')

    features['name'] = runner_json.get('name')
    features['draw'] = runner_json.get('draw')
    features['age'] = runner_json.get('age')
    features['sex'] = runner_json.get('sex')
    features['or'] = runner_json.get('ofr')
    features['rpr'] = runner_json.get('rpr')
    features['ts'] = runner_json.get('ts')
    features['lbs'] = runner_json.get('lbs')
    features['headgear'] = runner_json.get('headgear')
    features['last_run'] = runner_json.get('last_run')
    features['form'] = runner_json.get('form')

    trainer_stats = runner_json.get('stats', {}).get('trainer', {})
    features['trainer_id'] = str(runner_json.get('trainer_id')).strip()
    features['trainer_ovr_runs'] = trainer_stats.get('ovr_runs')
    features['trainer_ovr_wins'] = trainer_stats.get('ovr_wins')
    features['trainer_ovr_win_pct'] = clean_percent(trainer_stats.get('ovr_wins_pct'))
    features['trainer_ovr_profit'] = clean_float(trainer_stats.get('ovr_profit'))
    features['trainer_last_14_runs'] = trainer_stats.get('last_14_runs')
    features['trainer_last_14_wins'] = trainer_stats.get('last_14_wins')
    features['trainer_last_14_win_pct'] = clean_percent(trainer_stats.get('last_14_wins_pct'))
    features['trainer_last_14_profit'] = clean_float(trainer_stats.get('last_14_profit'))

    jockey_stats = runner_json.get('stats', {}).get('jockey', {})
    features['jockey_id'] = str(runner_json.get('jockey_id')).strip()
    features['jockey_ovr_runs'] = jockey_stats.get('ovr_runs')
    features['jockey_ovr_wins'] = jockey_stats.get('ovr_wins')
    features['jockey_ovr_win_pct'] = clean_percent(jockey_stats.get('ovr_wins_pct'))
    features['jockey_ovr_profit'] = clean_float(jockey_stats.get('ovr_profit'))
    features['jockey_last_14_runs'] = jockey_stats.get('last_14_runs')
    features['jockey_last_14_wins'] = jockey_stats.get('last_14_wins')
    features['jockey_last_14_win_pct'] = clean_percent(jockey_stats.get('last_14_wins_pct'))
    features['jockey_last_14_profit'] = clean_float(jockey_stats.get('last_14_profit'))

    features['comment'] = runner_json.get('comment')
    features['spotlight'] = runner_json.get('spotlight')

    features['num_prev_trainers'] = len(runner_json.get('prev_trainers') or [])
    features['num_prev_owners'] = len(runner_json.get('prev_owners') or [])

    features['rpr_rank'] = None
    features['or_rank'] = None
    features['rpr_zscore'] = None
    features['or_zscore'] = None

    return features

def inject_country_and_flatten(json_data):
    all_flattened = []

    for country, venues in json_data.items():
        for venue_name, races in venues.items():
            for off_time, race_data in races.items():
                race_context = {k: v for k, v in race_data.items() if k != "runners"}
                race_context['country'] = country

                gs_raw = race_context.get('GoingStick')
                race_context['GoingStick'] = float(gs_raw) if gs_raw else None

                race_class_parsed = parse_race_class(race_context.get('race_class', ''))
                race_context.update(race_class_parsed)

                for runner in race_data.get("runners", []):
                    features = extract_features_v2(runner, race_context)
                    features["race_id"] = race_context.get("race_id")
                    features["off_time"] = race_context.get("off_time")
                    features["venue"] = venue_name
                    features["race_datetime"] = f"{race_context.get('date')}T{race_context.get('off_time')}"

                    presence_fields = ['GoingStick', 'form', 'comment', 'spotlight', 'headgear']
                    features = add_presence_flags(features, presence_fields)

                    all_flattened.append(features)

    return pd.DataFrame(all_flattened)

def main(json_path, output_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    df = inject_country_and_flatten(raw_data)
    df = add_relative_features(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Flattened data saved to {output_path}")

# Example usage:
# main("data/raw/2025-03-26T08-58.json", "data/flattened/kemp_7pm.csv")
