import numpy as np
import pandas as pd

p4_data = np.load("data/PEMS04.npz")
p4_flow = p4_data['data']
p4_adj = pd.read_csv('data/PEMS04.csv', header=None)

p8_data = np.load("data/PEMS08.npz")
p8_flow = p8_data['data']
p8_adj = pd.read_csv('data/PEMS08.csv', header = None)

print(p4_flow.shape)
print(p8_flow.shape)

print(p4_adj.head())

print(p8_adj.head())


