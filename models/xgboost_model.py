"""
XGBoost Classifier (Ensemble Model)
"""

from xgboost import XGBClassifier

def build_model():
    """
    Returns an XGBoost Classifier
    """
    return XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
