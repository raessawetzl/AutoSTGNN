import numpy as np
import pandas as pd

# Configuration
#
# To build the adjacency matrix, we convert physical distances between 
# sensors into similarity weights using a Gaussian kernel formula:
# exp(-(distance^2) / sigma_squared). The closer two sensors are, the 
# higher their weight (closer to 1). The further apart, the lower (closer to 0).
#
# sigma_squared is automatically calculated from the distances in each 
# dataset's CSV file, so it adapts to each dataset rather than using a 
# hardcoded value.
#
# WEIGHT_THRESHOLD controls how sparse the final graph is — any sensor 
# pair whose weight falls below this value gets set to zero, meaning the 
# model treats them as unconnected


WEIGHT_THRESHOLD = 0.1 ##########################

DATASET_CONFIGS = [
    {"name": "PEMS04", "csv_path": "PEMS04.csv", "num_sensors": 307, "output_path": "PEMS04_adjacency.npz"},
    {"name": "PEMS08", "csv_path": "PEMS08.csv", "num_sensors": 170, "output_path": "PEMS08_adjacency.npz"},
]


def build_distance_matrix(distance_dataframe, num_sensors):
    """
    Takes the sensor distance CSV and builds a num_sensors x num_sensors 
    matrix where each cell holds the physical distance between two sensors.
    If two sensors have no listed connection in the CSV, that cell is set 
    to infinity — this gets converted to a weight of zero in the next step.
    The matrix is kept directed (not mirrored), because DCRNN treats 
    traffic flowing from A to B differently from B to A

    """
    distance_matrix = np.full((num_sensors, num_sensors), np.inf)

    row_index = 0
    num_rows = len(distance_dataframe)
    while row_index < num_rows:
        from_sensor = int(distance_dataframe.iloc[row_index]["from"])
        to_sensor = int(distance_dataframe.iloc[row_index]["to"])
        edge_cost = distance_dataframe.iloc[row_index]["cost"]
        distance_matrix[from_sensor, to_sensor] = edge_cost
        row_index = row_index + 1

    return distance_matrix


def apply_gaussian_kernel(distance_matrix, weight_threshold):
    """
    Converts physical distances into similarity weights using a
    Gaussian kernel, then zeroes out any weight below weight_threshold
    to keep the graph sparse. sigma_squared is the variance of the
    finite (i.e. actually listed) distances in the matrix.
    """
    finite_distances = distance_matrix[np.isfinite(distance_matrix)]
    sigma_squared = finite_distances.var()

    weight_matrix = np.exp(-(distance_matrix ** 2) / sigma_squared) # converts each distance into a similarity weight between 0 and 1 
    weight_matrix[weight_matrix < weight_threshold] = 0.0 #  Sensors with no edge (infinity distance)

    return weight_matrix, sigma_squared


def process_dataset(dataset_config, weight_threshold):
    dataset_name = dataset_config["name"]
    csv_path = dataset_config["csv_path"]
    num_sensors = dataset_config["num_sensors"]
    output_path = dataset_config["output_path"]

    csv_file = open(csv_path, "r")
    distance_dataframe = pd.read_csv(csv_file)
    csv_file.close()

    distance_matrix = build_distance_matrix(distance_dataframe, num_sensors)
    weight_matrix, sigma_squared = apply_gaussian_kernel(distance_matrix, weight_threshold)

    num_nonzero_edges = int((weight_matrix > 0).sum())

    output_file = open(output_path, "wb")
    np.savez(output_file, adjacency=weight_matrix)
    output_file.close()

    print("--- " + dataset_name + " ---")
    print("Sensors: " + str(num_sensors))
    print("Sigma squared (from listed distances): " + str(sigma_squared))
    print("Weight threshold: " + str(weight_threshold))
    print("Non-zero edges after thresholding: " + str(num_nonzero_edges))
    print("Saved adjacency matrix to " + output_path)
    print("")


def main():
    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        process_dataset(DATASET_CONFIGS[config_index], WEIGHT_THRESHOLD)
        config_index = config_index + 1


if __name__ == "__main__":
    main()