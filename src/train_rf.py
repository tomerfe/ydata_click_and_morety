# Modified version of train_rf.py
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix


def get_data():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    return df_train, df_test


def train(model_name: str, n_estimators: int, min_samples_leaf: int, max_depth: int, output_dir: str = "models"):
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    df_train, df_test = get_data()

    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test = df_test.drop(columns=['is_click'])
    y_test = df_test['is_click']

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight='balanced',
        criterion='entropy',
        max_features='log2',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_jobs=1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_test)

    # Calculate metrics
    f1_weighted = f1_score(y_test, y_train_pred, average='weighted')
    f1_binary = f1_score(y_test, y_train_pred, average='binary')

    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")

    # Create confusion matrix plot
    cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # Log metrics to wandb
    wandb.log({
        "F1_weighted": f1_weighted,
        "F1_binary": f1_binary,
        "model_name": model_name,
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": max_depth
    })

    # Save the model using joblib
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Also save model metadata
    model_metadata = {
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "parameters": {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth
        }
    }
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    joblib.dump(model_metadata, metadata_path)

    return model_path, f1_weighted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-m", "--model-name", default="RandomForest")
    parser.add_argument("-n", "--n-estimators", type=int, default=500)
    parser.add_argument("-l", "--min-samples-leaf", type=int, default=1)
    parser.add_argument("-d", "--max-depth", type=int, default=10)
    parser.add_argument("-o", "--output-dir", default="models")

    args = parser.parse_args()
    wandb.init(project="model-training", name=args.model_name, config=vars(args))
    train(args.model_name, args.n_estimators, args.min_samples_leaf, args.max_depth, args.output_dir)