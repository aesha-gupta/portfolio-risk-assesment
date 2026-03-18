import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame, is_training: bool = False) -> tuple:
    """
    Engineers financial risk features from a raw price DataFrame.
    Args:
        df: DataFrame containing 'Date', 'price', 'volume', 'ticker'
        is_training: If True, calculates the future 'risk_level' target variable.
    Returns:
        (features_df, target_series) 
        target_series is None if is_training=False.
    """
    # Sort by date just in case
    df = df.sort_values(by="Date").copy()
    
    # 1. Daily Returns (The foundation of risk modeling)
    df["daily_return"] = df["price"].pct_change()
    
    # 2. Rolling Volatility (30 days)
    # We use std dev of returns. (Multiplying by sqrt(252) would annualize it, but daily is fine for ML)
    df["volatility_30d"] = df["daily_return"].rolling(window=30).std()
    
    # 3. Momentum (30 days) - Current price vs 30-day moving average
    df["ma_30d"] = df["price"].rolling(window=30).mean()
    df["momentum_30d"] = (df["price"] - df["ma_30d"]) / df["ma_30d"]
    
    # 4. Maximum Drawdown (90 days)
    rolling_max_90d = df["price"].rolling(window=90, min_periods=1).max()
    df["drawdown"] = (df["price"] - rolling_max_90d) / rolling_max_90d
    df["max_drawdown_90d"] = df["drawdown"].rolling(window=90).min()
    
    # 5. Rolling Sharpe Ratio (30 days)
    # Annualized Sharpe: (Mean Daily Return / Daily Volatility) * sqrt(252 trading days)
    # Assuming risk-free rate is roughly 0% for this daily rolling metric
    df["mean_return_30d"] = df["daily_return"].rolling(window=30).mean()
    # Use np.where to avoid division by zero if volatility is exactly 0
    df["sharpe_ratio_30d"] = np.where(df["volatility_30d"] == 0, 0, 
                                     (df["mean_return_30d"] / df["volatility_30d"]) * np.sqrt(252))
    
    target = None
    
    if is_training:
        # SUPERVISED LEARNING TARGET GENERATION:
        # What makes it "High Risk"? A massive drop in the *future*.
        # We look ahead 252 trading days (1 year) to see the worst drawdown they would experience.
        
        # Reverse the dataframe to calculate forward-looking max drawdown
        df_reversed = df.iloc[::-1].copy()
        forward_max_price = df_reversed["price"].rolling(window=252, min_periods=1).max()
        forward_drawdown = (df_reversed["price"] - forward_max_price) / forward_max_price
        
        # Flip back to normal chronological order
        df["future_1yr_max_drawdown"] = forward_drawdown.iloc[::-1]
        
        # Rule: If the stock drops more than 25% in the next year, it was a "High Risk" time to buy.
        # If it drops between 10% and 25%, "Medium Risk". 
        # Otherwise "Low Risk".
        
        conditions = [
            (df["future_1yr_max_drawdown"] <= -0.25),
            (df["future_1yr_max_drawdown"] > -0.25) & (df["future_1yr_max_drawdown"] <= -0.10)
        ]
        choices = [2, 1] # 2=High, 1=Medium, 0=Low
        df["risk_level"] = np.select(conditions, choices, default=0)
        
        # Drop rows where we can't look 1 year ahead (the most recent year of data) 
        # because we don't know the true future drawdown yet.
        # And drop the first 90 days where rolling features are NaN.
        df.dropna(subset=["volatility_30d", "max_drawdown_90d", "future_1yr_max_drawdown"], inplace=True)
        
        target = df["risk_level"]
    else:
        # If evaluating user portfolio, we just drop the initial NaN rows from rolling windows
        df.dropna(subset=["volatility_30d", "max_drawdown_90d"], inplace=True)
    
    # Select final features
    feature_cols = ["daily_return", "volatility_30d", "momentum_30d", "max_drawdown_90d", "sharpe_ratio_30d"]
    features = df[feature_cols]
    
    # We return the modified df as well so the user can see their portfolio dates/prices
    return df, features, target

if __name__ == "__main__":
    from data.market_data_fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher()
    df_raw = fetcher.get_full_history("AAPL")
    
    print("\n--- Testing Feature Engineering (Training Mode) ---")
    df_full, features, target = prepare_features(df_raw, is_training=True)
    
    print("\nFeatures Head:")
    print(features.head())
    print("\nTarget Value Counts (0=Low, 1=Medium, 2=High Risk):")
    print(target.value_counts())
    
