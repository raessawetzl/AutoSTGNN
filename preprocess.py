import numpy as np

# Configuration: 
# 
# A run of consecutive zero flow readings at or above this length is
# treated as a sensor gap (and gets interpolated). Runs shorter than
# this are left untouched, since they arre assumed to represent genuine
# low traffic periods

GAP_THRESHOLD_HOURS = 2.0 ############# 

STEPS_PER_HOUR = 12  # 5 minute intervals -> 12 steps per hour
GAP_THRESHOLD_STEPS = int(GAP_THRESHOLD_HOURS * STEPS_PER_HOUR)

# Chronological split ratios for train/val/test 
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

DATASET_CONFIGS = [
    {"name": "PEMS04", "input_path": "PEMS04.npz", "output_path": "PEMS04_processed.npz"},
    {"name": "PEMS08", "input_path": "PEMS08.npz", "output_path": "PEMS08_processed.npz"},
]


def find_zero_runs(sensor_series):
    """
    Returns a list of (run_start, run_end) tuples 
    for every run of consecutive zero values in a single sensor's
    flow series.

    The output for one sensor might look something like [(150, 152), (4002, 4030), (11500, 11524)]
    """
    run_list = []
    series_length = len(sensor_series)
    current_index = 0
    while current_index < series_length:
        if sensor_series[current_index] == 0:
            run_start = current_index
            while current_index < series_length and sensor_series[current_index] == 0:
                current_index = current_index + 1
            run_end = current_index
            run_list.append((run_start, run_end))
        else:
            current_index = current_index + 1
    return run_list


def mark_long_runs_as_missing(flow_data, gap_threshold_steps):
    """
    flow_data: array of shape (num_timesteps, num_sensors)
    Returns a copy of flow_data where every zero-run of length >=
    gap_threshold_steps has been replaced with NaN, ready for
    interpolation. Also returns summary counts.
    """
    num_sensors = flow_data.shape[1]
    flow_data_with_gaps = flow_data.copy()

    total_runs_marked = 0
    total_steps_marked = 0

    sensor_index = 0
    while sensor_index < num_sensors:
        sensor_series = flow_data[:, sensor_index]
        zero_runs = find_zero_runs(sensor_series)

        run_index = 0
        while run_index < len(zero_runs):
            run_start, run_end = zero_runs[run_index]
            run_length = run_end - run_start
            if run_length >= gap_threshold_steps: # If a run meets or exceeds the threshold — likely sensor dropout — those positions get replaced with np.nan
                flow_data_with_gaps[run_start:run_end, sensor_index] = np.nan
                total_runs_marked = total_runs_marked + 1
                total_steps_marked = total_steps_marked + run_length
            run_index = run_index + 1

        sensor_index = sensor_index + 1

    return flow_data_with_gaps, total_runs_marked, total_steps_marked


def interpolate_missing_values(flow_data_with_gaps):
    """
    Receives the array where long zero-runs have been replaced with np.nan
    and its job is to fill those NaNs with estimated values.
    """
    num_timesteps = flow_data_with_gaps.shape[0]
    num_sensors = flow_data_with_gaps.shape[1]
    interpolated_flow_data = flow_data_with_gaps.copy()

    sensor_index = 0
    while sensor_index < num_sensors:
        sensor_series = interpolated_flow_data[:, sensor_index]
        is_missing = np.isnan(sensor_series)

        if is_missing.any():
            valid_indices = np.where(~is_missing)[0]
            all_indices = np.arange(num_timesteps)
            filled_series = np.interp(all_indices, valid_indices, sensor_series[valid_indices])
            interpolated_flow_data[:, sensor_index] = filled_series

        sensor_index = sensor_index + 1

    return interpolated_flow_data


def split_chronologically(flow_data, train_ratio, val_ratio):
    """
    Slices flow data along the time axis into train/val/test segments
    in chronological order  so test always represents
    the most recent period and train always the earliest
    """
    num_timesteps = flow_data.shape[0]
    train_end_index = int(round(num_timesteps * train_ratio))
    val_end_index = int(round(num_timesteps * (train_ratio + val_ratio)))

    train_data = flow_data[0:train_end_index] # This is what the model actually learns from — it sees these examples repeatedly and adjusts its weights based on them
    val_data = flow_data[train_end_index:val_end_index] # Validation is used during the CASH algorithm runs  to evaluate different hyperparameter configurations
    test_data = flow_data[val_end_index:num_timesteps] # Test is only used at the very end, to evaluate the final chosen model on unseen data and get a realistic estimate of how it will perform in production

    return train_data, val_data, test_data


def normalize_with_train_stats(train_data, val_data, test_data):
    """
    Computes a single global mean and standard deviation from the
    training data only, then applies that same scaling to train, val,
    and test. Using train-only stats avoids leaking information from
    the validation/test periods into the scaling.
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    normalized_train = (train_data - train_mean) / train_std
    normalized_val = (val_data - train_mean) / train_std
    normalized_test = (test_data - train_mean) / train_std

    return normalized_train, normalized_val, normalized_test, train_mean, train_std


def process_dataset(dataset_config, gap_threshold_steps, train_ratio, val_ratio):
    dataset_name = dataset_config["name"]
    input_path = dataset_config["input_path"]
    output_path = dataset_config["output_path"]

    input_file = open(input_path, "rb")
    raw_data = np.load(input_file)["data"]
    input_file.close()

    flow_data = raw_data[:, :, 0]

    flow_data_with_gaps, total_runs_marked, total_steps_marked = mark_long_runs_as_missing(
        flow_data, gap_threshold_steps
    )
    cleaned_flow_data = interpolate_missing_values(flow_data_with_gaps)

    train_data, val_data, test_data = split_chronologically(cleaned_flow_data, train_ratio, val_ratio)

    normalized_train, normalized_val, normalized_test, train_mean, train_std = normalize_with_train_stats(
        train_data, val_data, test_data
    )

    output_file = open(output_path, "wb")
    np.savez(
        output_file,
        train=normalized_train,
        val=normalized_val,
        test=normalized_test,
        mean=train_mean,
        std=train_std,
    )
    output_file.close()

    print("--- " + dataset_name + " ---")
    print("Gap threshold: " + str(gap_threshold_steps) + " steps (" + str(gap_threshold_steps / STEPS_PER_HOUR) + " hours)")
    print("Zero-runs marked as missing: " + str(total_runs_marked))
    print("Total timesteps interpolated: " + str(total_steps_marked))
    print("Train shape: " + str(normalized_train.shape))
    print("Val shape: " + str(normalized_val.shape))
    print("Test shape: " + str(normalized_test.shape))
    print("Train mean: " + str(train_mean) + ", train std: " + str(train_std))
    print("Saved processed data to " + output_path)
    print("")


def main():
    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        process_dataset(DATASET_CONFIGS[config_index], GAP_THRESHOLD_STEPS, TRAIN_RATIO, VAL_RATIO)
        config_index = config_index + 1


if __name__ == "__main__":
    main()