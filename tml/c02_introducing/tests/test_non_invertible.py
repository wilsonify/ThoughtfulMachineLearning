import numpy as np

from tml.c02_introducing.linear.get_list_to_matrix import get_list_to_matrix
from tml.c02_introducing.linear.linear_regression2 import MyLinearRegression


def test_non_invertible():
    # Second column is twice the first column
    # Third column is three times the first column
    x_matrix = get_list_to_matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9], ])
    y_matrix = get_list_to_matrix([[1], [2], [3]])
    is_invertible = np.linalg.det(x_matrix.T @ x_matrix) != 0
    assert not is_invertible
    mlr = MyLinearRegression()
    mlr.fit(x_matrix, y_matrix)
    assert mlr.beta.round(4).tolist() == [[0.0], [0.0714], [0.1429], [0.2143]]
