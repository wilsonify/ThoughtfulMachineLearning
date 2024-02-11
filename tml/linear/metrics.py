import numpy as np


def r_squared(y_true, y_pred):
    y_bar = np.mean(y_pred)
    deviations = y_true - y_bar
    residuals = y_pred - y_true
    total_sum_squares = np.sum(deviations.T @ deviations)
    residual_sum_squares = np.sum(residuals.T @ residuals)
    r_squared_value = 1 - (residual_sum_squares / total_sum_squares)
    return round(float(r_squared_value), 4)
