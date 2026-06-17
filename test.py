import numpy as np

data = np.load("data/PEMS04.npz")

flow = data['data']

print(flow.shape)