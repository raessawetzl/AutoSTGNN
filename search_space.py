

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
import numpy as np


def get_search_space(model_name):
    model_name = model_name.lower()

    cs = ConfigurationSpace(seed=42)

    # Shared training hyperparams (apply to all models)
    cs.add(Float('lr', bounds=(1e-4, 1e-2), log=True))
    cs.add(Categorical('batch_size', [16, 32, 64, 128]))

    if model_name == 'graphwavenet':
        cs.add(Categorical('hidden_size', [16, 32, 64]))
        cs.add(Categorical('ff_size', [128, 256, 512]))
        cs.add(Integer('n_layers', bounds=(4, 10)))
        cs.add(Integer('emb_size', bounds=(5, 20)))
        cs.add(Float('dropout', bounds=(0.0, 0.5)))

    elif model_name == 'dcrnn':
        cs.add(Categorical('hidden_size', [16, 32, 64]))
        cs.add(Integer('kernel_size', bounds=(1, 3)))
        cs.add(Integer('n_layers', bounds=(1, 3)))
        cs.add(Float('dropout', bounds=(0.0, 0.5)))

    elif model_name == 'stgcn':
        cs.add(Categorical('hidden_size', [32, 64, 128]))
        cs.add(Categorical('ff_size', [64, 128, 256]))
        cs.add(Integer('n_layers', bounds=(1, 4)))
        cs.add(Integer('temporal_kernel_size', bounds=(2, 5)))
        cs.add(Integer('spatial_kernel_size', bounds=(1, 3)))
        cs.add(Float('dropout', bounds=(0.0, 0.5)))

    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return cs