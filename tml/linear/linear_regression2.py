"""
same as linear_regression.py with a Class
"""
import json
import logging
from logging.config import dictConfig

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tml.linear.dataset import num_friends_good, daily_minutes_good
from tml.linear.get_list_to_matrix import get_list_to_matrix
from tml.linear.get_x_with_an_intercept import get_x_with_an_intercept, get_x_sans_intercept
from tml.linear.metrics import r_squared


def credible_interval_normal(y, confidence_level=0.95):
    y_mean = np.nanmean(y)
    y_stdev = np.nanstd(y)
    z_score = stats.norm.ppf((1 - confidence_level) / 2)
    margin_of_error = z_score * y_stdev
    interval_lower = y_mean - margin_of_error
    interval_upper = y_mean + margin_of_error
    return interval_lower, interval_upper


class MyLinearRegression:
    def __init__(self):
        self.beta = np.array([])
        self.bse = np.array([])
        self.x_nrows = 0.0
        self.x_mcols = 0.0
        self.y_nrows = 0.0
        self.y_mcols = 0.0
        self.degrees_of_freedom = 0.0
        self.rss = 0.0
        self.r2 = 0.0
        self.with_intercept = True
        self.y_low = 0.0
        self.y_high = 0.0
        self.x_mean = 0.0

    def fit(self, x_input, y_input, with_intercept=True):
        self.x_mean = np.nanmean(x_input)
        self.with_intercept = with_intercept
        x_original = x_input.copy()
        if with_intercept:
            x_input = get_x_with_an_intercept(x_input)
        else:
            x_input = get_x_sans_intercept(x_input)
        self.x_nrows = x_input.shape[0]
        self.x_mcols = x_input.shape[1]
        self.y_nrows = y_input.shape[0]
        self.y_mcols = y_input.shape[1]
        assert self.x_nrows == self.y_nrows
        self.degrees_of_freedom = self.x_nrows - self.x_mcols - self.y_mcols
        self.beta = np.linalg.pinv(x_input.T @ x_input) @ x_input.T @ y_input
        y_pred = self.predict(x_original)
        residuals = y_input - y_pred
        self.rss = np.sum(residuals.T @ residuals)
        sigma_squared_hat = self.rss / self.degrees_of_freedom
        var_beta_hat = np.linalg.pinv(x_input.T @ x_input) * sigma_squared_hat
        self.bse = np.sqrt(np.diagonal(var_beta_hat))
        self.r2 = r_squared(y_input, y_pred)
        self.y_low, self.y_high = credible_interval_normal(y_input)

    def predict(self, x_new):
        if self.with_intercept:
            x_new = get_x_with_an_intercept(x_new)
        else:
            x_new = get_x_sans_intercept(x_new)
        y_pred = x_new @ self.beta
        return y_pred

    def save(self, filename):
        attributes = {
            "beta": self.beta.tolist(),
            "bse": self.bse.tolist(),
            "x_nrows": self.x_nrows,
            "x_mcols": self.x_mcols,
            "y_nrows": self.y_nrows,
            "y_mcols": self.y_mcols,
            "degrees_of_freedom": self.degrees_of_freedom,
            "rss": self.rss,
            "r2": self.r2
        }
        with open(filename, "w") as file:
            json.dump(attributes, file)

    def load(self, filename):
        with open(filename, "r") as file:
            attributes = json.load(file)
            self.beta = np.array(attributes["beta"])
            self.bse = np.array(attributes["bse"])
            self.x_nrows = attributes["x_nrows"]
            self.x_mcols = attributes["x_mcols"]
            self.y_nrows = attributes["y_nrows"]
            self.y_mcols = attributes["y_mcols"]
            self.degrees_of_freedom = attributes["degrees_of_freedom"]
            self.rss = attributes["rss"]
            self.r2 = attributes["r2"]

    def plot_with_line(self, ax, x, y):
        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        ax.scatter(x, y)  # Plot the data points
        ax.set_xlabel('Number of Friends')
        ax.set_ylabel('Daily Minutes Spent on the Site')
        ax.set_title('Relationship between Number of Friends and Daily Minutes Spent')
        y_pred = self.predict(x)
        ax.plot(x, y_pred, color='red')  # Plot the regression line
        ax.scatter(x_mean, y_mean, color='black')

    def prediction_interval(self, x_new):
        """calculate confidence interval for prediction"""
        x_new_original = x_new.copy()
        if self.with_intercept:
            x_new = get_x_with_an_intercept(x_new_original)
        else:
            x_new = get_x_sans_intercept(x_new_original)

        alpha = 0.05
        half_alpha = alpha / 2.0
        quantile = 1 - half_alpha
        t_value = stats.t.ppf(quantile, self.degrees_of_freedom)
        margin_of_error = t_value * self.bse
        margin_of_error = margin_of_error.reshape(-1, 1)
        beta_low = self.beta - margin_of_error
        beta_high = self.beta + margin_of_error
        interval_lower = x_new @ beta_low
        interval_upper = x_new @ beta_high
        return interval_lower, interval_upper


def main():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    x_outer = num_friends_good_nda
    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)
    y_outer = daily_minutes_good_nda

    mlr = MyLinearRegression()
    mlr.fit(x_outer, y_outer)
    logging.info("intercept: %r", mlr.beta[0])
    logging.info("slope: %r", mlr.beta[1])
    logging.info("r-squared: %r", mlr.r2)

    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    mlr.plot_with_line(axs, num_friends_good_nda, daily_minutes_good_nda)
    # plt.show()


if __name__ == "__main__":
    dictConfig(dict(
        version=1,
        formatters={"simple": {"format": """%(asctime)s | %(name)s | %(lineno)s | %(levelname)s | %(message)s"""}},
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.DEBUG},
    ))
    main()
