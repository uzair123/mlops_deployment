from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    #  Train a classification model (Random Forest in this example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

     


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds


import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score


def model_performance_on_slices(model, X, y, categorical_feature_indices):
    """
    Outputs the performance of the model on slices of the data for each unique value of the categorical features.

    Inputs
    ------
    model : Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    y : np.array
        True labels.
    categorical_feature_indices : list of int
        List of column indices for the categorical features.

    Returns
    -------
    None, but prints performance metrics for each slice.
    """
    # Run predictions on the entire dataset
    preds = model.predict(X)

    for feature_idx in categorical_feature_indices:
        print(f"Performance metrics for slices of feature at column index: {feature_idx}")

        # Get the unique values of the feature at the given column index
        unique_values = np.unique(X[:, feature_idx])

        for value in unique_values:
            # Slice the data based on the current category value
            slice_mask = X[:, feature_idx] == value
            y_slice = y[slice_mask]
            preds_slice = preds[slice_mask]

            # Compute metrics for the slice
            precision = precision_score(y_slice, preds_slice, zero_division=1)
            recall = recall_score(y_slice, preds_slice, zero_division=1)
            fbeta = fbeta_score(y_slice, preds_slice, beta=1, zero_division=1)

            # Print the metrics for this slice
            print(f"  Category: {value}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1 Score: {fbeta:.4f}")
            print("\n")


