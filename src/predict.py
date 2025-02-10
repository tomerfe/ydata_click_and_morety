import argparse
import pandas as pd
import joblib
import os
import glob
import sys


# ----------------------------
# Stacking Predictor Wrapper
# ----------------------------
class StackingPredictor:
    def __init__(self, base_models, meta_model, variant='with_features'):
        """
        Parameters:
          base_models: dict with keys 'lightgbm', 'random_forest', 'xgboost', 'logistic_regression'
          meta_model: the stacking meta learner (loaded from file)
          variant: 'with_features' to concatenate original features with base predictions,
                   'without_features' to use only base predictions.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.variant = variant

    def predict(self, X):
        # Generate probability predictions from each base model
        preds = {}
        for key, model in self.base_models.items():
            preds[key] = model.predict_proba(X)[:, 1]

        # Build a DataFrame of base predictions
        df_meta = pd.DataFrame({
            'lgb_pred': preds['lightgbm'],
            'rf_pred': preds['random_forest'],
            'xgb_pred': preds['xgboost'],
            'lr_pred': preds['logistic_regression']
        })

        # Depending on the variant, either concatenate the original features or use only base predictions
        if self.variant == 'with_features':
            meta_features = pd.concat([X.reset_index(drop=True), df_meta.reset_index(drop=True)], axis=1)
        else:
            meta_features = df_meta

        # Return final predictions from the meta model
        return self.meta_model.predict(meta_features.values)


# ----------------------------
# Data Loading Function
# ----------------------------
def load_data():
    """
    Loads the preprocessed test data.
    Assumes that the preprocessing has saved a CSV file named "predict.csv" in the current directory.
    """
    return pd.read_csv("predict.csv")


# ----------------------------
# Base Models Loader
# ----------------------------
def load_base_models(model_dir):
    """
    Searches for and loads the base models from the given directory.
    Assumes filenames follow the patterns:
      - model_lightgbm_*.joblib
      - model_random_forest_*.joblib
      - model_xgboost_*.joblib
      - model_logistic_regression_*.joblib
    """
    base_models = {}
    patterns = {
        'lightgbm': os.path.join(model_dir, 'model_lightgbm_*.joblib'),
        'random_forest': os.path.join(model_dir, 'model_random_forest_*.joblib'),
        'xgboost': os.path.join(model_dir, 'model_xgboost_*.joblib'),
        'logistic_regression': os.path.join(model_dir, 'model_logistic_regression_*.joblib')
    }

    for key, pattern in patterns.items():
        files = glob.glob(pattern)
        if not files:
            print(f"Error: Could not find a file matching pattern: {pattern}")
            sys.exit(1)
        # Here we load the first file found; adjust the selection if needed.
        base_models[key] = joblib.load(files[0])
        print(f"Loaded base model '{key}' from {files[0]}")

    return base_models


# ----------------------------
# Main Prediction and Saving
# ----------------------------
def predict_and_save(meta_model_path, output_csv_path, variant):
    """
    Loads the meta model and base models, makes stacking predictions,
    and saves the output as a CSV containing only numeric values (no header).
    """
    # Load the preprocessed test data (assumed to be in the "data" directory)
    X_test = load_data()

    # Fill any missing values (using the mode) to be consistent with training
    X_test = X_test.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col, axis=0)

    # Load the meta (stacking) model from the specified path
    print(f"Loading meta model from {meta_model_path}")
    meta_model = joblib.load(meta_model_path)

    # Optionally, load meta model metadata if available
    metadata_path = meta_model_path.replace('.joblib', '_metadata.joblib')
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        print(f"Meta model metadata loaded: F1 weighted score = {metadata.get('f1_weighted', 0):.4f}")

    # Load base models from the "models" directory
    base_models = load_base_models("models")

    # Create the stacking predictor using the loaded models and the chosen variant
    predictor = StackingPredictor(base_models, meta_model, variant=variant)

    # Generate final predictions
    print("Making final stacking predictions...")
    y_pred = predictor.predict(X_test)

    # Save predictions as a CSV with no header and no index (only numbers)
    submission = pd.DataFrame(y_pred)
    submission.to_csv(output_csv_path, index=False, header=False)
    print(f"Predictions saved to {output_csv_path}")


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using stacking ensemble")
    parser.add_argument("-m", "--meta-model-path", required=True,
                        help="Path to the meta model file (e.g., models/meta_learner_with_features_*.joblib)")
    parser.add_argument("-o", "--output-csv-path", required=True,
                        help="Path to save predictions CSV (e.g., data/predictions.csv)")
    parser.add_argument("-v", "--variant", choices=['with_features', 'without_features'],
                        default='with_features',
                        help="Stacking variant: 'with_features' uses original features + base predictions; "
                             "'without_features' uses only base predictions")
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="Run preprocessing first (calls external preprocess.py)")

    args = parser.parse_args()

    # If the preprocess flag is given, run the external preprocessing script.
    # (Assumes that the preprocess script handles reading raw data and writing the preprocessed "predict.csv")
    if args.preprocess:
        import subprocess

        # Adjust the command and paths as needed.
        subprocess.run(["python", "../preprocess.py", "--mode", "predict",
                        "--train_path", "../data/train.csv", "--test_path", "../data/test.csv"])

    predict_and_save(args.meta_model_path, args.output_csv_path, args.variant)
