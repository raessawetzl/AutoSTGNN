import pandas as pd
import json


def results_to_excel(json_path, excel_path=None):
    """Convert a random_search results JSON file into a flat Excel spreadsheet."""
    with open(json_path, 'r') as f:
        results_log = json.load(f)

    rows = []
    for r in results_log:
        row = {
            'trial': r.get('trial'),
            'lr': r.get('lr'),
            'batch_size': r.get('batch_size'),
            'test_mae': r.get('test_mae'),
            'error': r.get('error'),
        }
        # flatten model_kwargs into separate columns
        for k, v in r.get('model_kwargs', {}).items():
            row[f'kw_{k}'] = v
        # flatten horizon_mae into separate columns
        for k, v in r.get('horizon_mae', {}).items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    if excel_path is None:
        excel_path = json_path.replace('.json', '.xlsx')

    df.to_excel(excel_path, index=False)
    print(f"Saved: {excel_path}")
    return df