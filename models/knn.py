"""
K-Nearest Neighbors (KNN) Classifier
"""

from sklearn.neighbors import KNeighborsClassifier

def build_model():
    """
    Returns a KNN Classifier
    """
    return KNeighborsClassifier(
        n_neighbors=5
    )
