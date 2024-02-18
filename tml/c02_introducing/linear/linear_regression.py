"""
all functions
"""
import logging
from logging.config import dictConfig

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from tml.linear.dataset import num_friends_good, daily_minutes_good


def get_list_to_matrix(x_list):
    assert isinstance(x_list, list)
    x_array = np.array(x_list)
    result = x_array
    if x_array.ndim == 0:  # Scalar value
        result = x_array.reshape(1, 1)
    if x_array.ndim == 1:  # 1D list
        result = x_array.reshape(-1, 1)
    return result


def get_x_with_an_intercept(x_nda):
    ones = np.ones(len(x_nda)).reshape(-1, 1)
    X_with_intercept = np.hstack((ones, x_nda))
    return X_with_intercept


def least_squares_fit(X, Y):
    # Implement least squares fit using matrix operations
    try:
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    except LinAlgError:
        beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return beta_hat


def least_squares_fit_hh(X, Y):
    x_nrows = X.shape[0]
    x_mcols = X.shape[1]
    y_nrows = Y.shape[0]
    y_mcols = Y.shape[1]
    assert x_nrows == y_nrows
    degrees_of_freedom = x_nrows - x_mcols - y_mcols
    confidence_level = 0.95
    tail = 1 - confidence_level
    right_tail = tail / 2
    right_quantile = 1 - right_tail
    t_crit = stats.t.ppf(right_quantile, degrees_of_freedom)

    # Implement least squares fit using matrix operations
    beta_hat = least_squares_fit(X, Y)
    se = std_error(X, Y)

    margin_of_error = t_crit * se[1]
    beta_hat[1] += margin_of_error
    x_bar = np.mean(X[1])
    y_bar = np.mean(Y)
    beta_hat[0] = y_bar - x_bar * beta_hat[1]
    return beta_hat


def least_squares_fit_ll(X, Y):
    x_nrows = X.shape[0]
    x_mcols = X.shape[1]
    y_nrows = Y.shape[0]
    y_mcols = Y.shape[1]
    assert x_nrows == y_nrows
    degrees_of_freedom = x_nrows - x_mcols - y_mcols
    confidence_level = 0.95
    tail = 1 - confidence_level
    right_tail = tail / 2
    right_quantile = 1 - right_tail
    t_crit = stats.t.ppf(right_quantile, degrees_of_freedom)

    # Implement least squares fit using matrix operations
    beta_hat = least_squares_fit(X, Y)
    se = std_error(X, Y)

    margin_of_error = t_crit * se[1]
    beta_hat[1] -= margin_of_error
    x_bar = np.mean(X[1])
    y_bar = np.mean(Y)
    beta_hat[0] = y_bar - x_bar * beta_hat[1]
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


def plot_with_line(ax, x, y):
    x_intercept = get_x_with_an_intercept(x)
    beta = least_squares_fit(x_intercept, y)
    beta_hh = least_squares_fit_hh(x_intercept, y)
    beta_ll = least_squares_fit_ll(x_intercept, y)
    se = std_error(x_intercept, y)
    # Plotting
    ax.scatter(x, y)
    ax.set_xlabel('Number of Friends')
    ax.set_ylabel('Daily Minutes Spent on the Site')
    ax.set_title('Relationship between Number of Friends and Daily Minutes Spent')
    # Plot the regression line
    y_pred = x_intercept @ beta
    ax.plot(x, y_pred, color='red')
    y_pred_hh = x_intercept @ beta_hh
    ax.plot(x, y_pred_hh, color='orange')
    y_pred_ll = x_intercept @ beta_ll
    ax.plot(x, y_pred_ll, color='green')

    ax.plot(np.mean(x), np.mean(x), color='black')
    ax.plot(np.mean(y), np.mean(y), color='black')


def main():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    X_with_intercept = get_x_with_an_intercept(num_friends_good_nda)

    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)
    Y = daily_minutes_good_nda

    beta = least_squares_fit(X_with_intercept, Y)
    y_pred = X_with_intercept @ beta
    logging.info("intercept: %r", beta[0])
    logging.info("slope: %r", beta[1])

    r_squared_f_vs_m = r_squared(Y, y_pred)
    logging.info("r-squared: %r", r_squared_f_vs_m)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    plot_with_line(ax, num_friends_good_nda, daily_minutes_good_nda)
    #plt.show()


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
    X_with_intercept = get_x_with_an_intercept(num_friends_good_nda)

    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)
    Y = daily_minutes_good_nda

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

    is_invertible = np.linalg.det(X.T @ X) != 0
    assert not is_invertible

    beta = least_squares_fit(X, Y)
    assert beta.round(4).tolist() == [[0.0714], [0.1429], [0.2143]]
