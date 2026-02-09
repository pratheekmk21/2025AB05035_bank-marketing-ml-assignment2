"""
Random Forest Classifier (Ensemble Model)
"""

from sklearn.ensemble import RandomForestClassifier

def build_model():
    """
    Returns a Random Forest Classifier
    """
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
