from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels import api as sm

from tml.c02_introducing.linear.dataset import num_friends_good, daily_minutes_good
from tml.c02_introducing.linear.get_list_to_matrix import get_list_to_matrix
from tml.c02_introducing.linear.get_x_with_an_intercept import get_x_with_an_intercept
from tml.c02_introducing.linear.linear_regression2 import MyLinearRegression


def test_least_squares_fit():
    num_friends_good_nda = get_list_to_matrix(num_friends_good)
    x_outer = num_friends_good_nda
    daily_minutes_good_nda = get_list_to_matrix(daily_minutes_good)
    y_outer = daily_minutes_good_nda

    # Least squares fit using matrix operations
    mlr = MyLinearRegression()
    mlr.fit(x_outer, y_outer)
    assert mlr.beta.round(4).tolist() == [[22.9476], [0.9039]]
    assert mlr.r2 == 0.3291
    assert mlr.bse.round(4).tolist() == [0.8478, 0.0913]

    # Fit model with scikit-learn
    lr = LinearRegression()
    lr.fit(X=num_friends_good_nda, y=daily_minutes_good_nda)

    assert lr.coef_.round(4).tolist() == [[0.9039]]
    assert lr.intercept_.round(4).tolist() == [22.9476]
    assert lr.n_features_in_ == 1
    assert lr.rank_ == 1
    assert lr.singular_.round(4).tolist() == [89.0164]
    lr_score = r2_score(y_true=y_outer, y_pred=lr.predict(num_friends_good_nda))
    assert round(lr_score, 4) == 0.3291

    # Confirming with statsmodels
    x_with_intercept = get_x_with_an_intercept(x_outer)
    model = sm.OLS(y_outer, x_with_intercept)
    results = model.fit()
    assert results.params.round(4).tolist() == [22.9476, 0.9039]
    assert round(results.rsquared, 4) == 0.3291
    assert results.bse.round(4).tolist() == [0.8457, 0.091]
