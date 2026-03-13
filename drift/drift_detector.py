def detect_risk_drift(df):

    """
    Detects portfolio risk drift and explains the reason.
    """

    # sort by client and month
    df = df.sort_values(by=["client_id", "month"])

    # calculate changes
    df["volatility_change"] = df.groupby("client_id")["volatility"].diff()
    df["concentration_change"] = df.groupby("client_id")["concentration"].diff()
    df["equity_change"] = df.groupby("client_id")["equity_pct"].diff()

    # initialize columns
    df["risk_drift"] = False
    df["drift_reason"] = ""

    # iterate through rows
    for i in df.index:

        reasons = []

        if abs(df.loc[i, "volatility_change"]) > 5:
            reasons.append("Volatility change")

        if abs(df.loc[i, "concentration_change"]) > 20:
            reasons.append("Concentration change")

        if abs(df.loc[i, "equity_change"]) > 25:
            reasons.append("Equity exposure change")

        if reasons:
            df.loc[i, "risk_drift"] = True
            df.loc[i, "drift_reason"] = ", ".join(reasons)

    return df