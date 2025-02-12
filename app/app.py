import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,balanced_accuracy_score
)
from xgboost import XGBClassifier

# -------------------------
# Helper Functions
# -------------------------

def concatenate_columns(row):
    """Concatenates several categorical columns into a single string."""
    return f"{row['gender']}|{row['age_level']}|{row['user_group_id']}|{row['user_depth']}"

@st.cache_data(show_spinner=False)
def feature_engineering(df):
    """Performs various feature engineering steps on the dataframe."""
    # Map campaign_id to webpage_id and vice versa
    campaign_id_to_webpage_id = df.groupby('campaign_id')['webpage_id'].first().to_dict()
    df.fillna({'webpage_id': df['campaign_id'].map(campaign_id_to_webpage_id)}, inplace=True)
    
    webpage_id_to_campaign_id = df.groupby('webpage_id')['campaign_id'].first().to_dict()
    df.fillna({'campaign_id': df['webpage_id'].map(webpage_id_to_campaign_id)}, inplace=True)

    # Create normalized counts
    df['webpage_id_count'] = df.groupby('user_id')['webpage_id'].transform('count') / len(df)
    df['campaign_id_impression_count'] = df.groupby('campaign_id')['user_id'].transform('count') / len(df)
    
    # Fill missing values for several columns
    df.fillna({'product': 'X', 'product_category_2': 1000.0}, inplace=True)
    df.fillna({
        'user_group_id': df['user_group_id'].max() + 1,
        'gender': 'Other',
        'age_level': df['age_level'].max() + 1,
        'user_depth': df['user_depth'].max() + 1
    }, inplace=True)
    
    df['user_group_id'] = df['user_group_id'].astype(int)
    df['age_level'] = df['age_level'].astype(int)
    df['user_depth'] = df['user_depth'].astype(int)
    
    # Create user_type and bag_of_products features
    df['user_type'] = df.apply(concatenate_columns, axis=1)
    df['bag_of_products'] = df.groupby('user_id')['product'].transform(lambda x: ' '.join(x.unique()))
    df['bag_of_products'] = df['bag_of_products'].astype('category').cat.codes
    df['user_type'] = df['user_type'].astype('category').cat.codes
    df['var_1'] = df['var_1'].astype('category').cat.codes

    df.fillna({'city_development_index': df['city_development_index'].mode()[0]}, inplace=True)

    # Process DateTime features
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['hour'] = df['DateTime'].dt.hour
    df['time'] = df['hour'] // 3  # 8 bins (0-7)
    df['day_of_week'] = df['DateTime'].dt.weekday
    df['day'] = np.where(df['day_of_week'] < 5, 'weekday', 'weekend')
    df.drop(columns=['hour', 'day_of_week'], inplace=True)

    # Convert categorical columns to numerical codes
    cat_columns = [
        'gender', 'product', 'campaign_id', 'webpage_id',
        'product_category_1', 'product_category_2', 'city_development_index',
        'bag_of_products', 'day', 'var_1', 'user_type'
    ]
    for col in cat_columns:
        df.fillna({col: 'other'}, inplace=True)
        df[col] = df[col].astype('category').cat.codes

    return df

@st.cache_resource(show_spinner=False)
def get_model():
    """Loads the pre-trained model from disk."""
    with open('../models/xgb_classifier_model.sav', 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluates the model using an adjustable threshold for classification
       and displays various Plotly charts including cool visualizations for
       classification metrics (now including balanced accuracy) and model explainability."""
    # Get predicted probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    # Apply the threshold slider to generate class predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # -----------------------
    # Classification Metrics Visualization with Plotly Indicators
    # -----------------------
    # Create subplots for six indicators.
    fig_metrics = make_subplots(
        rows=1, cols=6,
        subplot_titles=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Balanced Accuracy"],
        specs=[[{"type": "indicator"}] * 6]
    )
    
    # Each metric is multiplied by 100 to display as a percentage.
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=acc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=1)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=prec * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=2)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=rec * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=3)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=f1 * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=4)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=auc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=5)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=balanced_acc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=6)
    
    fig_metrics.update_layout(height=200, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_metrics)
    
    # -----------------------
    # ROC Curve
    # -----------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"AUC = {auc:.3f}",
                                 line=dict(color='darkorange')))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random',
                                 line=dict(dash='dash', color='navy')))
    fig_roc.update_layout(title="ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc)
    
    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_cm.update_layout(title="Confusion Matrix",
                         xaxis_title="Predicted Label",
                         yaxis_title="True Label")
    st.plotly_chart(fig_cm)
    
    # -----------------------
    # Classification Report
    # -----------------------
    st.text("Classification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
    
    # -----------------------
    # Histogram of Predicted Probabilities
    # -----------------------
    fig_hist = px.histogram(x=y_proba, nbins=50, title="Histogram of Predicted Probabilities")
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold = {threshold:.2f}")
    fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Frequency")
    st.plotly_chart(fig_hist)


# -------------------------
# Main App
# -------------------------

def main():
    st.set_page_config(page_title="CTR Click Prediction Evaluation", layout="wide")
    st.title("CTR Click Prediction Evaluation App")
    
    # Create two tabs: one for uploading a test set & scoring, and one for full model evaluation.
    tab_upload, tab_eval = st.tabs(["Upload & Score", "Model Evaluation"])
    
    # -------------------------
    # Tab 1: Upload & Score
    # -------------------------
    with tab_upload:
        st.header("Upload Test Set & Download Predictions")
        st.markdown("""
        **Features:**
        - Upload your test set (CSV file) from your local machine.
        - The app will produce predicted probabilities (scores) for each sample.
        - Download a CSV file containing only the scores.
        """)
        
        uploaded_test_file = st.file_uploader("Upload Test Set CSV", type=["csv"])
        if uploaded_test_file is not None:
            try:
                # Read the uploaded CSV file; parse 'DateTime' if present.
                test_df = pd.read_csv(uploaded_test_file, parse_dates=["DateTime"])
            except Exception:
                test_df = pd.read_csv(uploaded_test_file)
            st.write("Preview of Uploaded Test Set:")
            st.dataframe(test_df.head())
            
            # Apply feature engineering.
            with st.spinner("Applying feature engineering..."):
                test_df = feature_engineering(test_df)
                # Remove the raw DateTime column if it exists.
                if 'DateTime' in test_df.columns:
                    test_df = test_df.drop(columns=['DateTime'])
            
            X_test = test_df.copy()  # All columns are used for prediction.
            
            with st.spinner("Loading model and making predictions..."):
                model = get_model()
                # Generate predicted probabilities for the positive class.
                predicted_scores = model.predict_proba(X_test)[:, 1]
            
            st.success("Predictions completed!")
            
            # Prepare CSV output: one column with scores, no header, no index.
            output_csv = io.StringIO()
            pd.DataFrame(predicted_scores).to_csv(output_csv, header=False, index=False)
            csv_output = output_csv.getvalue()
            
            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="../../../../../Downloads/predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Please upload a test set CSV file.")
    
    # -------------------------
    # Tab 2: Model Evaluation
    # -------------------------
    with tab_eval:
        st.subheader("Preprocessing & Evaluation")
        # For demonstration, we load our internal train/test data.
        # (In your use case, you might adapt this section accordingly.)
        with st.spinner("Loading and preprocessing data..."):
            # Here we simulate by loading internal datasets.
            # In practice, your test set might not have labels.
            # The following assumes your test set for evaluation includes labels.
            train_df = pd.read_csv('../data/train_dataset_full.csv', parse_dates=['DateTime'])
            test_df = pd.read_csv('../data/X_test_1st.csv', parse_dates=['DateTime'])
            test_labels = pd.read_csv('../data/y_test_1st.csv', header=None).values.reshape(-1)
            test_df['is_click'] = test_labels
            
            # Mark the source of each record before concatenation.
            train_df['is_test'] = 0
            test_df['is_test'] = 1
            
            # Combine train and test, then apply feature engineering.
            df_all = pd.concat([train_df, test_df], ignore_index=True)
            df_all = feature_engineering(df_all)
            if 'DateTime' in df_all.columns:
                df_all.drop(columns=['DateTime'], inplace=True)
            df_test = df_all[df_all['is_test'] == 1].copy()
            X_test_eval = df_test.drop(columns=['is_click', 'is_test'])
            y_test = df_test['is_click']
        st.success("Data loaded and preprocessed successfully!")
        
        with st.spinner("Loading model..."):
            model = get_model()
        st.success("Model loaded!")
        
        # Sidebar options for evaluation.
        threshold = st.sidebar.slider("Probability Threshold", min_value=0.0, max_value=1.0,
                                      value=0.5, step=0.01)
        show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
        
        # Evaluate the model.
        evaluate_model(model, X_test_eval, y_test, threshold)
        
        # Feature Importance (if available).
        if show_feature_importance and hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            feature_names = X_test_eval.columns
            df_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df_importance.sort_values("Importance", ascending=True, inplace=True)
            fig_imp = px.bar(df_importance, x="Importance", y="Feature", orientation="h",
                             title="Feature Importances")
            st.plotly_chart(fig_imp)
        
        
if __name__ == '__main__':
    main()
