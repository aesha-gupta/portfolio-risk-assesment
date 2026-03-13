from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from xgboost import XGBClassifier


def train_models(X, y):

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Random Forest Model
    # -------------------------
    rf_model = RandomForestClassifier(random_state=42)

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print("\nRandom Forest Accuracy:", rf_accuracy)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))
    print("\nRandom Forest Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))

    # -------------------------
    # XGBoost Model
    # -------------------------
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    xgb_model.fit(X_train, y_train)

    xgb_predictions = xgb_model.predict(X_test)

    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    print("XGBoost Accuracy:", xgb_accuracy)
    print("\nXGBoost Classification Report:") 
    print(classification_report(y_test, xgb_predictions)) 
    print("\nXGBoost Confusion Matrix:") 
    print(confusion_matrix(y_test, xgb_predictions)) 

    return rf_model, xgb_model