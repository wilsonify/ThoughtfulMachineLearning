import logging
from logging.config import dictConfig

import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

num_friends_good = [
    49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12, 11, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]

daily_minutes_good = [
    68.77, 51.25, 52.08, 38.36, 44.54, 57.13, 51.4, 41.42, 31.22, 34.76, 54.01, 38.79, 47.59, 49.1, 27.66, 41.03,
    36.73, 48.65, 28.12, 46.62, 35.57, 32.98, 35, 26.07, 23.77, 39.73, 40.57, 31.65, 31.21, 36.32, 20.45, 21.93,
    26.02, 27.34, 23.49, 46.94, 30.5, 33.8, 24.23, 21.4, 27.94, 32.24, 40.57, 25.07, 19.42, 22.39, 18.42, 46.96,
    23.72, 26.41, 26.97, 36.76, 40.32, 35.02, 29.47, 30.2, 31, 38.11, 38.18, 36.31, 21.03, 30.86, 36.07, 28.66,
    29.08, 37.28, 15.28, 24.17, 22.31, 30.17, 25.53, 19.85, 35.37, 44.6, 17.23, 13.47, 26.33, 35.02, 32.09, 24.81,
    19.33, 28.77, 24.26, 31.98, 25.73, 24.86, 16.28, 34.51, 15.23, 39.72, 40.8, 26.06, 35.76, 34.76, 16.13, 44.04,
    18.03, 19.65, 32.62, 35.59, 39.43, 14.18, 35.24, 40.13, 41.82, 35.45, 36.07, 43.67, 24.61, 20.9, 21.9, 18.79,
    27.61, 27.21, 26.61, 29.77, 20.59, 27.53, 13.82, 33.2, 25, 33.1, 36.65, 18.63, 14.87, 22.2, 36.81, 25.53, 24.62,
    26.25, 18.21, 28.08, 19.42, 29.79, 32.8, 35.99, 28.32, 27.79, 35.88, 29.06, 36.28, 14.1, 36.63, 37.49, 26.9,
    18.58, 38.48, 24.48, 18.95, 33.55, 14.24, 29.04, 32.51, 25.63, 22.22, 19, 32.73, 15.16, 13.9, 27.2, 32.01,
    29.27, 33, 13.74, 20.42, 27.32, 18.23, 35.35, 28.48, 9.08, 24.62, 20.12, 35.26, 19.92, 31.02, 16.49, 12.16,
    30.7, 31.22, 34.65, 13.13, 27.51, 33.2, 31.57, 14.1, 33.42, 17.44, 10.12, 24.42, 9.82, 23.39, 30.93, 15.03,
    21.67, 31.09, 33.29, 22.61, 26.89, 23.48, 8.38, 27.81, 32.35, 23.84,
]


def get_list_to_matrix(x_list):
    assert isinstance(x_list, list)
    x_array = np.array(x_list)
    result = x_array
    if x_array.ndim == 0:  # Scalar value
        result = x_array.reshape(1, 1)
    if x_array.ndim == 1:  # 1D list
        result = x_array.reshape(-1, 1)
    return result


def get_y_as_a_column():
    y_nda = get_list_to_matrix(daily_minutes_good)
    return y_nda


def get_x_sans_intercept():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    return num_friends_good_nda


def get_x_with_an_intercept():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    ones = np.ones(len(num_friends_good)).reshape(-1, 1)
    X_with_intercept = np.hstack((ones, num_friends_good_nda))
    return X_with_intercept


def least_squares_fit(X, Y):
    # Implement least squares fit using matrix operations
    try:
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    except LinAlgError:
        beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return beta_hat


def r_squared(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    y_bar = np.mean(y_pred)
    deviations = y_true - y_bar
    residuals = y_pred - y_true
    total_sum_squares = float(np.sum(deviations.T @ deviations))
    residual_sum_squares = float(np.sum(residuals.T @ residuals))
    r_squared_value = 1 - (residual_sum_squares / total_sum_squares)
    result = round(float(r_squared_value), 4)
    return result


def std_error(X, y_true):
    x_nrows = X.shape[0]
    x_mcols = X.shape[1]
    y_nrows = y_true.shape[0]
    y_mcols = y_true.shape[1]
    assert x_nrows == y_nrows
    degrees_of_freedom = x_nrows - x_mcols - y_mcols
    beta = least_squares_fit(X, y_true)
    y_pred = X @ beta
    y_pred = y_pred.reshape(-1, 1)
    residuals = y_true - y_pred
    residual_sum_of_squares = float(np.sum(residuals.T @ residuals))
    sigma_squared_hat = residual_sum_of_squares / degrees_of_freedom
    var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat
    std_error_value = np.sqrt(np.diagonal(var_beta_hat))
    return std_error_value


def main():
    X_with_intercept = get_x_with_an_intercept()
    Y = get_y_as_a_column()

    beta = least_squares_fit(X_with_intercept, Y)
    logging.info("intercept: %r", beta[0])
    logging.info("slope: %r", beta[1])
    # r_squared_f_vs_m = r_squared(alpha_, beta_, X[:, 0], Y)
    # logging.info("r-squared: %r", r_squared_f_vs_m)


def test_main():
    dictConfig(dict(
        version=1,
        formatters={"simple": {"format": """%(asctime)s | %(name)s | %(lineno)s | %(levelname)s | %(message)s"""}},
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.DEBUG},
    ))
    main()


def test_least_squares_fit():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)

    X_with_intercept = get_x_with_an_intercept()
    Y = get_y_as_a_column()

    # Least squares fit using matrix operations
    beta = least_squares_fit(X_with_intercept, Y)
    assert beta.round(4).tolist() == [[22.9476], [0.9039]]
    y_pred = X_with_intercept @ beta
    y_pred = y_pred.reshape(-1, 1)
    r_squared_score = r_squared(Y, y_pred)
    assert r_squared_score == 0.3291
    se = std_error(X_with_intercept, Y)
    assert se.round(4).tolist() == [0.8478, 0.0913]

    # Fit model with scikit-learn

    lr = LinearRegression()
    lr.fit(X=num_friends_good_nda, y=daily_minutes_good_nda)

    assert lr.coef_.round(4).tolist() == [[0.9039]]
    assert lr.intercept_.round(4).tolist() == [22.9476]
    assert lr.n_features_in_ == 1
    assert lr.rank_ == 1
    assert lr.singular_.round(4).tolist() == [89.0164]
    lr_score = r2_score(y_true=Y, y_pred=lr.predict(num_friends_good_nda))
    assert round(lr_score, 4) == 0.3291

    # Confirming with statsmodels
    model = sm.OLS(Y, X_with_intercept)
    results = model.fit()
    assert results.params.round(4).tolist() == [22.9476, 0.9039]
    assert round(results.rsquared, 4) == 0.3291
    assert results.bse.round(4).tolist() == [0.8457, 0.091]


def test_non_intertable():
    # Second column is twice the first column
    # Third column is three times the first column
    X = get_list_to_matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9], ])
    Y = get_list_to_matrix([[1], [2], [3]])

    # Calculate X.T @ X
    X_T_X = X.T @ X

    # Check if X.T @ X is invertible
    is_invertible = np.linalg.det(X_T_X) != 0
    assert not is_invertible

    beta = least_squares_fit(X, Y)
    assert beta.round(4).tolist() == [[0.0714], [0.1429], [0.2143]]
