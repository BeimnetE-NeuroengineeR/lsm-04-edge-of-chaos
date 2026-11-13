
import numpy as np


def generate_input_signal(n_samples=800, seed=42):
    """
    Generate a reproducible uniform random input signal.

    Parameters
    ----------
    n_samples : int
        Number of input samples.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Random signal in the range [-1, 1].
    """
    rng = np.random.default_rng(seed)

    signal = rng.uniform(
        low=-1.0,
        high=1.0,
        size=n_samples
    )

    return signal
