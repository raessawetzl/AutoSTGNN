from tsl.datasets import MetrLA, PemsBay
metr = MetrLA(root='./data')
print('METR-LA loaded')
print('Shape:', metr.dataframe().shape)