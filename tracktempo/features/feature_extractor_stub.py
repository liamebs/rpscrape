
"""
Feature Extractor Stub for Horse Racing Runner JSON

This module defines a function to convert a runner JSON entry into a structured feature dictionary.
It handles core, categorical, numeric, engineered, and optional NLP/embedding features.
"""

def extract_features(runner_json, race_context):
    """
    Convert a single runner JSON and race context into a flat dictionary of features.
    
    Args:
        runner_json (dict): Runner-level JSON object
        race_context (dict): Race-level context (shared across all runners in same race)
    
    Returns:
        dict: Flattened feature dict for model input
    """

    features = {}

    # --- RACE CONTEXT ---
    features['course'] = race_context.get('course')
    features['country'] = race_context.get('country')
    features['going'] = race_context.get('going')
    features['GoingStick'] = race_context.get('GoingStick')
    features['distance_f'] = race_context.get('distance_f')
    features['field_size'] = race_context.get('field_size')
    features['race_class'] = race_context.get('race_class')
    features['type'] = race_context.get('type')

    # --- RUNNER CORE FEATURES ---
    features['horse_id'] = runner_json.get('horse_id')
    features['name'] = runner_json.get('name')
    features['draw'] = runner_json.get('draw')
    features['age'] = runner_json.get('age')
    features['sex'] = runner_json.get('sex')
    features['or'] = runner_json.get('ofr')
    features['rpr'] = runner_json.get('rpr')          # ⚠️ Check source time
    features['ts'] = runner_json.get('ts')            # ⚠️ Check source time
    features['lbs'] = runner_json.get('lbs')
    features['headgear'] = runner_json.get('headgear')
    features['last_run'] = runner_json.get('last_run')
    features['form'] = runner_json.get('form')        # ⚠️ May need cleaning

    # --- TRAINER STATS (from 'stats' block) ---
    trainer_stats = runner_json.get('stats', {}).get('trainer', {})
    features['trainer_id'] = runner_json.get('trainer_id')
    features['trainer_ovr_runs'] = trainer_stats.get('ovr_runs')
    features['trainer_ovr_wins'] = trainer_stats.get('ovr_wins')
    features['trainer_ovr_win_pct'] = trainer_stats.get('ovr_wins_pct')
    features['trainer_ovr_profit'] = trainer_stats.get('ovr_profit')
    features['trainer_last_14_runs'] = trainer_stats.get('last_14_runs')
    features['trainer_last_14_wins'] = trainer_stats.get('last_14_wins')
    features['trainer_last_14_win_pct'] = trainer_stats.get('last_14_win_pct')
    features['trainer_last_14_profit'] = trainer_stats.get('last_14_profit')

    # --- JOCKEY STATS (from 'stats' block) ---
    jockey_stats = runner_json.get('stats', {}).get('jockey', {})
    features['jockey_id'] = runner_json.get('jockey_id')
    features['jockey_ovr_runs'] = jockey_stats.get('ovr_runs')
    features['jockey_ovr_wins'] = jockey_stats.get('ovr_wins')
    features['jockey_ovr_win_pct'] = jockey_stats.get('ovr_win_pct')
    features['jockey_ovr_profit'] = jockey_stats.get('ovr_profit')
    features['jockey_last_14_runs'] = jockey_stats.get('last_14_runs')
    features['jockey_last_14_wins'] = jockey_stats.get('last_14_wins')
    features['jockey_last_14_win_pct'] = jockey_stats.get('last_14_win_pct')
    features['jockey_last_14_profit'] = jockey_stats.get('last_14_profit')

    # --- NLP OPTIONAL ---
    features['comment'] = runner_json.get('comment')
    features['spotlight'] = runner_json.get('spotlight')

    # --- RELATIONAL FLAT COUNTS ---
    features['num_prev_trainers'] = len(runner_json.get('prev_trainers') or [])
    features['num_prev_owners'] = len(runner_json.get('prev_owners') or [])

    # --- ENGINEERED PLACEHOLDERS (to be computed later) ---
    features['rpr_rank'] = None
    features['or_rank'] = None
    features['rpr_zscore'] = None
    features['or_zscore'] = None

    return features