"""
Logistic Regression model for Bank Marketing Dataset

This file defines the Logistic Regression classifier
used in Assignment 2.
"""

from sklearn.linear_model import LogisticRegression

def build_model():
    """
    Returns a Logistic Regression model
    """
    return LogisticRegression(max_iter=1000)
