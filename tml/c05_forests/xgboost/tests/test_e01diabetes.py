from io import StringIO

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from tml.c05_forests.xgboost.e01diabetes import (
    load_data,
    split_data,
    train_model,
    evaluate_model
)

# Sample dataset as a string for testing purposes
SAMPLE_DATA = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
"""


def test_load_data():
    """
    Test that the dataset is loaded correctly.
    """
    dataset = load_data(StringIO(SAMPLE_DATA))
    assert dataset.shape == (9, 9)


def test_split_data():
    """
    Test that the dataset is split into training and testing sets correctly.
    """
    dataset = np.array([
        [6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
        [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
        [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1],
        [1, 89, 66, 23, 94, 28.1, 0.167, 21, 0],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33, 1],
    ])
    X_train, X_test, y_train, y_test = split_data(dataset, test_size=0.4, seed=7)

    assert X_train.shape == (3, 8)  # 3 samples in training
    assert X_test.shape == (2, 8)  # 2 samples in testing
    assert len(y_train) == 3
    assert len(y_test) == 2


def test_train_model():
    """
    Test that the model trains correctly.
    """
    X_train = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
                        [8, 183, 64, 0, 0, 23.3, 0.672, 32]])
    y_train = np.array([1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, XGBClassifier)
    assert model.n_classes_ == 2  # Ensure binary classification


def test_evaluate_model():
    """
    Test that the model evaluation works correctly.
    """
    # Mock data
    y_test = np.array([1, 0, 1])
    predictions = np.array([1, 0, 1])

    # Mock model
    class MockModel:
        def predict(self, X):
            return predictions

    mock_model = MockModel()
    accuracy = evaluate_model(mock_model, None, y_test)
    assert accuracy == accuracy_score(y_test, predictions)
    assert accuracy == 1.0  # 100% accuracy


def test_end_to_end():
    """
    End-to-end test of the workflow using the sample dataset.
    """
    # Load data
    dataset = load_data(StringIO(SAMPLE_DATA))

    # Split data
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0  # Ensure accuracy is a valid percentage
