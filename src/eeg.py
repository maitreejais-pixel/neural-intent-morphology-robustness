import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = [
    "Forward",
    "Left",
    "Right",
    "Stop",
]


def generate_synthetic_eeg(
    n_trials=576,
    n_channels=22,
    fs=250,
    duration=4.0,
    seed=7,
):
    """
    Generate synthetic EEG trials for four locomotion intentions.

    Classes:
        0 = Forward
        1 = Left
        2 = Right
        3 = Stop

    The signal contains class-dependent oscillatory components
    in the sensorimotor channels, embedded in Gaussian noise.
    """

    rng = np.random.default_rng(seed)

    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    # Background EEG-like noise.
    X = rng.normal(
        0.0,
        1.0,
        (n_trials, n_channels, n_samples),
    )

    # Balanced class labels.
    y = np.repeat(
        np.arange(len(CLASSES)),
        n_trials // len(CLASSES),
    )

    rng.shuffle(y)

    # Synthetic sensorimotor channel groups.
    sensorimotor = np.arange(8)
    central = np.array([3, 4, 5, 6])

    for i, cls in enumerate(y):

        # Variable temporal envelope.
        onset = rng.uniform(1.0, 2.0)
        width = rng.uniform(0.8, 1.4)

        envelope = np.exp(
            -0.5
            * ((t - onset) / width) ** 2
        )

        # Class-dependent spatial/frequency pattern.
        if cls == 0:
            # Forward
            channels = central
            freq = rng.uniform(10, 13)

        elif cls == 1:
            # Left
            channels = sensorimotor[:4]
            freq = rng.uniform(9, 13)

        elif cls == 2:
            # Right
            channels = sensorimotor[4:8]
            freq = rng.uniform(9, 13)

        else:
            # Stop
            channels = sensorimotor[2:6]
            freq = rng.uniform(18, 25)

        # Simulate occasional absence of the class-specific
        # oscillatory component.
        if rng.random() >= 0.14:

            phase = rng.uniform(
                0,
                2 * np.pi,
            )

            component = (
                np.sin(
                    2 * np.pi * freq * t
                    + phase
                )
                * envelope
            )

            for ch in channels:
                X[i, ch] -= (
                    rng.uniform(0.35, 0.9)
                    * component
                )

        # Channel-specific amplitude variation.
        X[i] *= rng.uniform(
            0.75,
            1.25,
            n_channels,
        )[:, None]

        # Measurement noise.
        X[i] += rng.normal(
            0.0,
            0.12,
            X[i].shape,
        )

    return (
        X.astype(np.float32),
        y.astype(int),
        fs,
    )


def log_bandpower_features(
    X,
    fs,
    low=8,
    high=30,
):
    """
    Extract log bandpower features from each EEG channel.

    Welch's method is used to estimate the power spectral
    density, followed by integration over the requested
    frequency band.
    """

    features = []

    for trial in X:

        row = []

        for channel in trial:

            frequencies, power = welch(
                channel,
                fs=fs,
                nperseg=min(
                    500,
                    len(channel),
                ),
            )

            mask = (
                (frequencies >= low)
                & (frequencies <= high)
            )

            bandpower = np.trapezoid(
                power[mask],
                frequencies[mask],
            )

            row.append(
                np.log(
                    bandpower + 1e-12
                )
            )

        features.append(row)

    return np.asarray(
        features,
        dtype=np.float64,
    )


def evaluate_decoder(
    X,
    y,
    fs=250,
    seed=7,
):
    """
    Evaluate the synthetic EEG intention decoder.

    Pipeline:
        log-bandpower features
        -> StandardScaler
        -> LDA

    Evaluation:
        5-fold stratified cross-validation.
    """

    features = log_bandpower_features(
        X,
        fs,
    )

    pipeline = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    predictions = cross_val_predict(
        pipeline,
        features,
        y,
        cv=cv,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
            )
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ),
    }

    # Fit the final decoder on the complete dataset.
    pipeline.fit(
        features,
        y,
    )

    return pipeline, metrics


def make_frozen_intention_stream(
    n_steps,
    seed=101,
):
    """
    Generate a reproducible intention sequence.

    The base sequence cycles through:
        Forward -> Left -> Right -> Stop

    A small stochastic perturbation rate is included
    to prevent the stream from being perfectly periodic.
    """

    rng = np.random.default_rng(seed)

    stream = np.array(
        [
            i % len(CLASSES)
            for i in range(n_steps)
        ],
        dtype=int,
    )

    for i in range(1, n_steps):

        if rng.random() < 0.015:
            stream[i] = rng.integers(
                0,
                len(CLASSES),
            )

    return stream

