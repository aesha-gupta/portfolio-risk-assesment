def detect_risk_drift(df):
    """
    Detects portfolio risk drift over time for a given ticker.
    Flags days where the risk profile fundamentally shifted.
    """
    # Ensure it's sorted by date
    df = df.sort_values(by="Date").copy()
    
    # Calculate daily changes in our risk features
    df["volatility_change"] = df["volatility_30d"].diff()
    df["momentum_change"] = df["momentum_30d"].diff()
    df["drawdown_change"] = df["max_drawdown_90d"].diff()

    # initialize columns
    df["risk_drift"] = False
    df["drift_reason"] = ""

    # iterate through rows
    for i in df.index:
        reasons = []

        # Volatility spike
        if abs(df.loc[i, "volatility_change"]) > 0.02: # 2% jump in 30d std dev is massive
            reasons.append("Volatility Spike")

        # Momentum collapse
        if df.loc[i, "momentum_change"] < -0.10: # 10% drop relative to moving average
            reasons.append("Momentum Collapse")

        # Drawdown acceleration
        if df.loc[i, "drawdown_change"] < -0.05: # Drawdown worsened by 5% in a day
            reasons.append("Severe Drawdown")

        if reasons:
            df.loc[i, "risk_drift"] = True
            df.loc[i, "drift_reason"] = ", ".join(reasons)

    return df