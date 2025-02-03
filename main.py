from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.evaluate_model import evaluate_model
import pandas as pd

# Load and preprocess data
df, df_train, x_test = load_data("data/train_dataset_full.csv", "data/x_test_1.csv")
df = preprocess_data(df)

# Feature engineering
df_train, x_test = create_features(df_train, x_test)

# Train models
X_train, y_train = df_train.drop(columns=["is_click"]), df_train["is_click"]
models = train_models(X_train, y_train)

# Evaluate a model
evaluate_model(models["RandomForest"], X_train, y_train)

print("Pipeline completed!")
