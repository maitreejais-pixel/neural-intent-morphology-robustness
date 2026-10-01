from src.eeg import generate_synthetic_eeg, evaluate_decoder


def main():
    # Generate the reproducible synthetic EEG dataset.
    X, y, fs = generate_synthetic_eeg(
        n_trials=576,
        n_channels=22,
        fs=250,
        duration=4.0,
        seed=7,
    )

    # Evaluate the LDA decoder using 5-fold stratified CV.
    _, metrics = evaluate_decoder(
        X,
        y,
        fs,
        seed=7,
    )

    print("=" * 60)
    print("Synthetic EEG Decoder Evaluation")
    print("=" * 60)

    print(
        f"Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("=" * 60)


if __name__ == "__main__":
    main()

