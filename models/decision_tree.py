"""
Decision Tree Classifier for Bank Marketing Dataset
"""

from sklearn.tree import DecisionTreeClassifier

def build_model():
    """
    Returns a Decision Tree Classifier
    """
    return DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
