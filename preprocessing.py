import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    x_test = pd.read_csv(test_path)
    
    # Add a flag to distinguish train and test data
    df_train['is_train'] = 1
    x_test['is_train'] = 0

    # Combine datasets for consistent preprocessing
    df = pd.concat([df_train, x_test], ignore_index=True)
    
    return df, df_train, x_test

def preprocess_data(df):
    df.drop(['product_category_2'], axis=1, inplace=True)
    
    labels = ['gender', 'age_level', 'user_group_id', 'city_development_index']
    df[labels] = df.groupby('user_id')[labels].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    
    df['total_sessions_per_user'] = df.groupby('user_id')['session_id'].transform('count')
    df['day_binary'] = (df['day'] > 4.5).astype(int)
    
    return df

if __name__ == "__main__":
    df, df_train, x_test = load_data("data/train_dataset_full.csv", "data/x_test_1.csv")
    df = preprocess_data(df)
    print("Data preprocessing completed!")
