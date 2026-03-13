import pandas as pd

def prepare_features(df):
    #Dividing the dataset into features and target variable 
    df['concentration']=df[['equity_pct', 'bonds_pct', 'crypto_pct']].max(axis=1) 
    features=df[["equity_pct", "bonds_pct", "crypto_pct", "volatility", "return","concentration"]]
    risk_mapping={
        "low":0,"medium":1,"high":2                          
    }

    target=df['risk_level'].str.lower().map(risk_mapping)          # perfomed normalization on target variable and mapped to numerical values (TARGET VARIABLES MAY CONTAIN INCONSISTENCIES LIKE Medium or medium, so we convert to lowercase before mapping)
    df["dominant_asset"] = df[["equity_pct","bonds_pct","crypto_pct"]].idxmax(axis=1)   

    return features,target
