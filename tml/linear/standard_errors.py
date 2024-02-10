import pandas as pd
import numpy as np
from sklearn import linear_model
randn = np.random.randn

X = pd.DataFrame(randn(10,3), columns=['X1','X2','X3'])
y = pd.DataFrame(randn(10,1), columns=['Y'])        

model = linear_model.LinearRegression()
model.fit(X=X, y=y)
# -

# ## show scikit-learn results

print("from scikit-learn")
print(model.intercept_)
print(model.coef_)

# ## reproduce scikit-learn results with linear algebra

N = len(X)
p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X.values

beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
print(beta_hat)

# ## compute standard errors of the parameter estimates

y_hat = model.predict(X)
residuals = y.values - y_hat
residual_sum_of_squares = residuals.T @ residuals
sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
for p_ in range(p):
    standard_error = var_beta_hat[p_, p_] ** 0.5
    print(f"SE(beta_hat[{p_}]): {standard_error}")

# ## confirm with `statsmodels`

import statsmodels.api as sm
ols = sm.OLS(y.values, X_with_intercept)
ols_result = ols.fit()
print(ols_result.summary())


