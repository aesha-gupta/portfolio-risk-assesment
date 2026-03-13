from sklearn.ensemble import IsolationForest


def detect_anomalies(X):

    """
    Detects anomalous portfolios using Isolation Forest.
    """

    # create model
    model = IsolationForest(contamination=0.05, random_state=42)
    #The contamination parameter represents the expected proportion of anomalies in the dataset. Increasing it causes the Isolation Forest model to classify more observations as anomalies because the decision threshold becomes less strict.

    # train model
    model.fit(X)

    # predict anomalies
    predictions = model.predict(X)

    return predictions


#The supervised models predict the risk level of a portfolio, while the anomaly detection model identifies portfolios whose asset allocation patterns significantly deviate from typical portfolios, even if their risk level appears low.