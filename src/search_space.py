import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_shared_config_space():
    """
    Hyperparameters that are genuinely shared across AGCRN, DCRNN, and STGCN.
    dropout is NOT included here because AGCRN has no dropout layers.
    hidden_units maps to hidden_dim (AGCRN), rnn_units (DCRNN), or the
    channel dimension (STGCN) — the trainer is responsible for passing
    it to the right constructor argument per model.
    """
    cs = CS.ConfigurationSpace(seed=42)

    cs.add([
        CSH.UniformFloatHyperparameter(
            'learning_rate', lower=1e-4, upper=1e-2, log=True
        ),
        CSH.UniformIntegerHyperparameter(
            'num_layers', lower=1, upper=4
        ),
        CSH.UniformIntegerHyperparameter(
            'hidden_units', lower=32, upper=256
        ),
        CSH.CategoricalHyperparameter(
            'batch_size', choices=[16, 32, 64]
        ),
        CSH.UniformFloatHyperparameter(
            'weight_decay', lower=1e-5, upper=1e-3, log=True
        ),
    ])
    return cs


def get_agcrn_config_space():
    """
    AGCRN-specific additions:
    - cheb_k: Chebyshev polynomial order for adaptive graph convolution
    - embed_dim: node embedding dimension, controls expressiveness of
      the learned adaptive adjacency matrix (E @ E.T)
    No dropout — AGCRN has no dropout layers in its architecture.
    """
    cs = get_shared_config_space()
    cs.add([
        CSH.CategoricalHyperparameter(
            'cheb_k', choices=[1, 2, 3]
        ),
        CSH.CategoricalHyperparameter(
            'embed_dim', choices=[8, 10, 16, 32]
        ),
    ])
    return cs


def get_dcrnn_config_space():
    """
    DCRNN-specific additions:
    - diffusion_steps: maps to max_diffusion_step in DCGRUCell, controls
      how many random-walk steps are used to aggregate spatial context.
      The original paper uses 2. No Chebyshev order — DCRNN uses
      random-walk diffusion, not Chebyshev polynomials.
    No dropout — confirmed from dcrnn.py and dcrnn_cell.py, no Dropout
    layers anywhere in the architecture.
    """
    cs = get_shared_config_space()
    cs.add([
        CSH.UniformIntegerHyperparameter(
            'diffusion_steps', lower=1, upper=5
        ),
    ])
    return cs


def get_stgcn_config_space():
    """
    STGCN-specific additions:
    - cheb_k: Chebyshev filter order for spatial graph conv (ks in stgcn.py)
    - kernel_size: temporal conv kernel size (kt in stgcn.py), must be odd
      for symmetric padding — choices restricted to odd values only
    - dropout: applied inside each st_conv_block after layer norm,
      confirmed from stgcn.py line 76
    """
    cs = get_shared_config_space()
    cs.add([
        CSH.UniformIntegerHyperparameter(
            'cheb_k', lower=2, upper=5
        ),
        CSH.CategoricalHyperparameter(
            'kernel_size', choices=[3, 5, 7]
        ),
        CSH.UniformFloatHyperparameter(
            'dropout', lower=0.0, upper=0.5
        ),
    ])
    return cs