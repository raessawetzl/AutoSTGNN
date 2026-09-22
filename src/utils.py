import json
import numpy as np
import pandas as pd


def to_native(value):
    """Converts NumPy values to standard Python types for JSON output."""
    if isinstance(value, dict):
        native_dict = {}
        for key in value:
            native_dict[key] = to_native(value[key])
        return native_dict
    if isinstance(value, (list, tuple)):
        native_list = []
        item_index = 0
        while item_index < len(value):
            native_list.append(to_native(value[item_index]))
            item_index = item_index + 1
        return native_list
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_best_val_mae(trainer):
    """
    Returns the best validation MAE from a training run
    """
    checkpoint_callback = getattr(trainer, 'checkpoint_callback', None)
    if checkpoint_callback is not None:
        best_score = getattr(checkpoint_callback, 'best_model_score', None)
        if best_score is not None:
            return float(best_score)
    return float(trainer.callback_metrics.get('val_mae', float('inf')))


def save_json(results_log, out_path):
    """Saves the results as a JSON file."""
    output_file = open(out_path, 'w')
    json.dump(to_native(results_log), output_file, indent=2)
    output_file.close()


def results_to_excel(json_path, excel_path=None):
    """Converts a JSON results file into an Excel spreadsheet.
    """
    input_file = open(json_path, 'r')
    results_log = json.load(input_file)
    input_file.close()

    if isinstance(results_log, dict):
        results_log = [results_log]

    rows = []
    record_index = 0
    while record_index < len(results_log):
        record = results_log[record_index]
        row = {}
        for key in record:
            value = record[key]
            if isinstance(value, dict):
                # Nested values are added as separate columns
                prefix = 'kw_' if key == 'model_kwargs' else ''
                for sub_key in value:
                    col_name = prefix + sub_key if prefix else sub_key
                    row[col_name] = value[sub_key]
            else:
                row[key] = value
        rows.append(row)
        record_index = record_index + 1

    df = pd.DataFrame(rows)

    if excel_path is None:
        excel_path = json_path.replace('.json', '.xlsx')

    df.to_excel(excel_path, index=False)
    print("Saved: " + excel_path)
    return df
