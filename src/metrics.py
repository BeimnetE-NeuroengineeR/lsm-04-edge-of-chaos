
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def compute_memory_profile(
    reservoir_states,
    input_signal,
    max_delay=20,
    train_fraction=0.60,
    alpha=10.0
):
    """
    Measure how well reservoir states reconstruct
    previous input values.

    Uses a chronological train/test split.
    """

    capacities = []

    for delay in range(1, max_delay + 1):

        # Current reservoir state predicts past input.
        X = reservoir_states[delay:]
        y = input_signal[:-delay]

        split = int(
            len(y) * train_fraction
        )

        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        score = r2_score(
            y_test,
            predictions
        )

        capacities.append(float(score))

    # Standard positive-only memory-capacity sum.
    total_capacity = float(
        np.sum(
            np.maximum(capacities, 0.0)
        )
    )

    mean_r2 = float(
        np.mean(capacities)
    )

    return (
        list(range(1, max_delay + 1)),
        capacities,
        total_capacity,
        mean_r2
    )
