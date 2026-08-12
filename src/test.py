from weatherbench import WeatherBench1

ds = WeatherBench1(
    root='./data/weatherbench1_test',
    variable='geopotential_500',  # confirmed name from the search results
    resolution='5.625deg',
    start_time='2015-01-01',
    end_time='2015-01-31',
)
print(ds)

connectivity = ds.get_connectivity(
    method='distance',
    threshold=0.4,
    include_self=False,
    normalize_axis=1,
    layout='edge_index'
)
edge_index, edge_weight = connectivity
n_edges = edge_index.shape[1]
print(f"Edges: {n_edges}")
print(f"Nodes: {ds.n_nodes}")
print(f"Edges per node: {n_edges / ds.n_nodes:.1f}")