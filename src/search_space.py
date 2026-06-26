import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def get_shared_config_space():
    """
    hyperparameters shared across STGCN, DCRNN and Graph WaveNet.
    all three models use this as their base search space.
    training budget (num_epochs) is controlled by BOHB externally 
    """
            
    cs = CS.ConfigurationSpace(seed=42) # search space container 

    # learning rate
    learning_rate = CSH.UniformFloatHyperparameter(
        name='learning_rate',
        lower=1e-4,
        upper=1e-2,
        log=True # BOHB will search more densely at the smaller end than the larger end
    )

    # number of graph convolution layers
    num_layers = CSH.UniformIntegerHyperparameter(
        name='num_layers',
        lower=1,
        upper=4
    )

    # hidden units
    hidden_units = CSH.UniformIntegerHyperparameter(
        name='hidden_units',
        lower=32,
        upper=256
    )

    # dropout rate
    dropout = CSH.UniformFloatHyperparameter(
        name='dropout',
        lower=0.0,
        upper=0.5
    )

    # batch size
    batch_size = CSH.CategoricalHyperparameter(
        name='batch_size',
        choices=[16, 32, 64]
    )

    # for L2 regularisation
    weight_decay = CSH.UniformFloatHyperparameter(
        name='weight_decay',
        lower=1e-5,
        upper=1e-3,
        log=True
    )

    # add hyperparameters to the search space
    cs.add([
        learning_rate,
        num_layers,
        hidden_units,
        dropout,
        batch_size, 
        weight_decay
    ])

    return cs

def get_stgcn_config_space() -> CS.ConfigurationSpace:
    """STGCN search space = shared params + Chebyshev filter order"""
    cs = get_shared_config_space()

    cs.add(
        CSH.UniformIntegerHyperparameter('K_cheb', lower=1, upper=5)
    )

    return cs