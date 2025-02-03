import pandas as pd

def create_features(df_train, x_test):
    high_correlation_pairs = [(82320, 1734), (98970, 6970)]
    
    for camp, web in high_correlation_pairs:
        df_train[f'campaign_{camp}_webpage_{web}'] = ((df_train['campaign_id'] == camp) & (df_train['webpage_id'] == web)).astype(int)
        x_test[f'campaign_{camp}_webpage_{web}'] = ((x_test['campaign_id'] == camp) & (x_test['webpage_id'] == web)).astype(int)
    
    return df_train, x_test

if __name__ == "__main__":
    df_train = pd.read_csv("data/train_processed.csv")
    x_test = pd.read_csv("data/x_test_processed.csv")
    
    df_train, x_test = create_features(df_train, x_test)
    print("Feature engineering completed!")
