"""
utils.py

json -> excel conversion for search results.

flattens a random_search / bohb results json log into a flat spreadsheet:
nested dict fields (model_kwargs, horizon_mae, etc.) each get their own
columns, alongside every top-level scalar field.

usage:
    from utils import results_to_excel
    results_to_excel('search_results/stgcn_electricity_RS_20260921.json')

config:
    json_path   - path to the results json log
    excel_path  - output .xlsx path (defaults to json_path with .xlsx extension)
"""

import json
import pandas as pd


def results_to_excel(json_path, excel_path=None):
    """flatten a results json log into a dataframe and write it to excel."""
    with open(json_path, 'r') as f:
        results_log = json.load(f)

    rows = []
    for r in results_log:
        row = {}
        for key, value in r.items():
            if isinstance(value, dict):
                # flatten nested dicts (model_kwargs, horizon_mae, test_results, etc.)
                prefix = 'kw_' if key == 'model_kwargs' else ''
                for sub_key, sub_value in value.items():
                    col_name = f'{prefix}{sub_key}' if prefix else sub_key
                    row[col_name] = sub_value
            else:
                row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    if excel_path is None:
        excel_path = json_path.replace('.json', '.xlsx')

    df.to_excel(excel_path, index=False)
    print(f"Saved: {excel_path}")
    return df