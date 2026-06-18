import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# reusable search space 
def get_config_space():
    cs = CS.ConfigurationSpace() # search space container 

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

    # add hyperparameters to the search space
    cs.add_hyperparameters([
        learning_rate,
        num_layers,
        hidden_units,
        dropout,
        batch_size
    ])

    return cs
