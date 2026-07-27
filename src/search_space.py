import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
def get_config_space(model_name):
    cs = CS.ConfigurationSpace(seed=42)

    # shared across all models
    cs.add([
        CSH.UniformFloatHyperparameter("lr",           lower=1e-4, upper=1e-2, log=True),
        CSH.UniformFloatHyperparameter("weight_decay", lower=1e-5, upper=1e-3, log=True),
        CSH.UniformIntegerHyperparameter("rnn_units",  lower=32,   upper=128),
        CSH.UniformIntegerHyperparameter("num_layers", lower=1,    upper=3),
    ])

    if model_name == "AGCRN":
        cs.add([
            CSH.UniformIntegerHyperparameter("embed_dim", lower=4,   upper=20),
            CSH.UniformIntegerHyperparameter("cheb_k",    lower=2,   upper=4),
        ])

    elif model_name == "DCRNN":
        cs.add([
            CSH.UniformIntegerHyperparameter("max_diffusion_step", lower=1, upper=3),
        ])

    elif model_name == "STGCN":
        cs.add([
            CSH.UniformIntegerHyperparameter("K",       lower=2,  upper=5),
            CSH.UniformFloatHyperparameter("dropout",   lower=0.0, upper=0.5),
        ])

    return cs