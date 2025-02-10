# train_ada.py
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import joblib
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix


def get_data():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    return df_train, df_test


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    return cm


def train(model_name: str, n_estimators: int, learning_rate: float, max_depth: int, output_dir: str = "models"):
    # Initialize wandb
    wandb.init(
        project="adaboost-classifier",
        name=model_name,
        config={
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    df_train, df_test = get_data()

    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test = df_test.drop(columns=['is_click'])
    y_test = df_test['is_click']

    base_estimator = DecisionTreeClassifier(max_depth=max_depth)

    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary = f1_score(y_test, y_pred, average='binary')

    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")

    # Plot and save confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred, output_dir)

    # Log metrics to wandb
    wandb.log({
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "confusion_matrix": wandb.Image(os.path.join(output_dir, 'confusion_matrix.png'))
    })

    # Save the model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    # Log model to wandb
    wandb.save(model_path)

    wandb.finish()

    return model, f1_weighted, f1_binary, cm


def main():
    parser = argparse.ArgumentParser(description='Train AdaBoost Classifier')
    parser.add_argument('--model-name', type=str, default='adaboost_model',
                        help='Name of the model')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators')
    parser.add_argument('--learning-rate', type=float, default=1.0,
                        help='Learning rate')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Maximum depth of base estimator')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models and artifacts')

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()