from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    f1 = f1_score(y_val, y_pred)
    print(f"F1 Score: {f1}")
    
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    from model_training import train_models
    import pandas as pd
    
    df_train = pd.read_csv("data/train_final.csv")
    y_train = df_train.pop("is_click")
    
    models = train_models(df_train, y_train)
    evaluate_model(models["RandomForest"], df_train, y_train)
