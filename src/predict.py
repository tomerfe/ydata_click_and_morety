# Modified version of predict.py
import argparse
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier


def load_data():
    """Loads the preprocessed test data."""
    return pd.read_csv("data/predict.csv")


def predict_and_save(model_path, output_csv_path):
    """Loads model, predicts, and saves output."""
    # Load test data
    X_test = load_data()

    # Ensure consistency with training features
    X_test = X_test.apply(lambda col: col.fillna(col.mode()[0]), axis=0)

    # Load trained model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load model metadata if it exists
    metadata_path = model_path.replace('.joblib', '_metadata.joblib')
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        print(f"Model metadata loaded: F1 weighted score = {metadata['f1_weighted']:.4f}")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Save predictions
    submission = pd.DataFrame({'is_click': y_pred})
    submission.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("-m", "--model-path", required=True, help="Path to trained model")
    parser.add_argument("-o", "--output-csv-path", required=True, help="Path to save predictions")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Run preprocessing first")

    args = parser.parse_args()

    # Run preprocessing if requested
    if args.preprocess:
        import subprocess

        subprocess.run(["python", "preprocess.py", "--mode", "predict"])

    predict_and_save(args.model_path, args.output_csv_path)