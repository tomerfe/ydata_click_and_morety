import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import joblib
import xgboost as xgb
import os
from sklearn.metrics import f1_score, confusion_matrix


def get_data():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    return df_train, df_test


def train(model_name: str, max_depth: int, learning_rate: float, n_estimators: int, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    df_train, df_test = get_data()

    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test = df_test.drop(columns=['is_click'])
    y_test = df_test['is_click']

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective='binary:logistic',
        scale_pos_weight=1,  # Equivalent to class_weight='balanced'
        booster='gbtree',
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary = f1_score(y_test, y_pred, average='binary')

    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    wandb.log({
        "F1_weighted": f1_weighted,
        "F1_binary": f1_binary,
        "model_name": model_name,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators
    })

    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model_path, f1_weighted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Trainer")
    parser.add_argument("-m", "--model-name", default="XGBoost")
    parser.add_argument("-n", "--n-estimators", type=int, default=100)
    parser.add_argument("-d", "--max-depth", type=int, default=6)
    parser.add_argument("-r", "--learning-rate", type=float, default=0.1)
    parser.add_argument("-o", "--output-dir", default="models")

    args = parser.parse_args()
    wandb.init(project="model-training", name=args.model_name, config=vars(args))
    train(args.model_name, args.max_depth, args.learning_rate, args.n_estimators, args.output_dir)