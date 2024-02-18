from matplotlib import pyplot as plt
from statsmodels import api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table

from tml.linear.dataset import num_friends_good, daily_minutes_good
from tml.linear.get_list_to_matrix import get_list_to_matrix
from tml.linear.get_x_with_an_intercept import get_x_with_an_intercept
from tml.linear.linear_regression2 import MyLinearRegression


def test_interval():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    x_nda = num_friends_good_nda
    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)
    y_nda = daily_minutes_good_nda
    x_with_intercept = get_x_with_an_intercept(x_nda)
    model = sm.OLS(y_nda, x_with_intercept)
    results = model.fit()
    print(results.summary())
    assert results.params.round(4).tolist() == [22.9476, 0.9039]
    assert round(results.rsquared, 4) == 0.3291
    assert results.bse.round(4).tolist() == [0.8457, 0.091]

    predstd, interval_lower, interval_upper = wls_prediction_std(results)

    assert predstd.shape == (203,)
    assert interval_lower.shape == (203,)
    assert interval_upper.shape == (203,)

    predictions = results.get_prediction(x_with_intercept)
    predictions.summary_frame(alpha=0.05)
    simple_table, measures_data, column_names = summary_table(results, alpha=0.05)
    print(simple_table)

    dep_var_population = measures_data[:, 1]
    predicted_value = measures_data[:, 2]
    mean_ci_95_low = measures_data[:, 4]
    mean_ci_95_upp = measures_data[:, 5]
    cooks_distance = measures_data[:, 11]

    mlr = MyLinearRegression()
    mlr.fit(x_nda, y_nda)
    low, high = mlr.prediction_interval(x_nda)
    assert low.shape == y_nda.shape
    assert high.shape == y_nda.shape

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].scatter(x_nda, y_nda, label="y_nda")
    axs[0].scatter(x_nda, dep_var_population, label="dep_var_population")
    axs[0].plot(x_nda, predicted_value, label="predicted_value")
    axs[0].plot(x_nda, mean_ci_95_low, label="mean_ci_95_low")
    axs[0].plot(x_nda, predicted_value, label="predicted_value")
    axs[0].plot(x_nda, mean_ci_95_upp, label="mean_ci_95_upp")

    axs[1].scatter(x_nda, y_nda, label="y_nda")
    axs[1].plot(x_nda, low, label="low")
    axs[1].plot(x_nda, mlr.predict(x_nda), label="pred")
    axs[1].plot(x_nda, high, label="high")
    # axs[1].scatter(x_nda, cooks_distance, label="cooks_distance")

    for ax in axs.flat:
        ax.legend()
    #plt.show()
    #assert low.round(4).tolist() == mean_ci_95_low.round(4).tolist()
    #assert high.round(4).tolist() == mean_ci_95_upp.round(4).tolist()


def ols_quantile(m, x_in, q):
    a = q * 2
    if q > 0.5:
        a = 2 * (1 - q)
    predictions = m.get_prediction(x_in)
    frame = predictions.summary_frame(alpha=a)
    if q > 0.5:
        return frame.obs_ci_upper
    return frame.obs_ci_lower
