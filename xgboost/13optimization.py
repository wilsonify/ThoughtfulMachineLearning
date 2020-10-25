"""
using xgboost with bayesian optimzation
"""
import numpy as np
import pandas as pd
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy.optimize import differential_evolution

print(sorted(sklearn.metrics.SCORERS.keys()))
DATA_TRAIN_PATH = 'data/train.csv.zip'
DATA_TEST_PATH = 'data/test.csv.zip'

train = pd.read_csv(DATA_TRAIN_PATH)
train = train.sample(1000)

train_labels = [int(v[-1]) - 1 for v in train.target.values]
labels = np.array(train_labels)
train_ids = train.id.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
train_nda = train.values

test = pd.read_csv(DATA_TEST_PATH)
test = test.sample(100)
test_ids = test.id.values
test = test.drop('id', axis=1)
test_nda = test.values

bounds_dict = {
    'max_depth': (5, 20),
    'learning_rate': (0.01, 1),
    'n_estimators': (20, 1000),
    'gamma': (1., 0.01),
    'min_child_weight': (2, 10),
    'max_delta_step': (0, 3),
    'subsample': (0.7, 0.85),
    'colsample_bytree': (0.5, 0.99)
}
bounds_list = [val for val in bounds_dict.values()]


def xgb_cost_function(input_array):
    (max_depth,
     learning_rate,
     n_estimators,
     gamma,
     min_child_weight,
     max_delta_step,
     subsample,
     colsample_bytree) = input_array
    model = XGBClassifier(
        objective="multi:softprob",
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=-1
    )
    cross_validation_scores = cross_val_score(
        estimator=model,
        X=train,
        y=labels,
        groups=None,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        verbose=0,
        fit_params=None,
        pre_dispatch='2*n_jobs',
        error_score=np.nan
    )
    log_loss = -cross_validation_scores.mean()

    return log_loss


def xgboostcv(
        max_depth,
        learning_rate,
        n_estimators,
        gamma,
        min_child_weight,
        max_delta_step,
        subsample,
        colsample_bytree):
    model = XGBClassifier(
        objective="multi:softprob",
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=-1
    )
    cross_validation_scores = cross_val_score(
        estimator=model,
        X=train,
        y=labels,
        groups=None,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        verbose=0,
        fit_params=None,
        pre_dispatch='2*n_jobs',
        error_score=np.nan
    )
    log_loss = -cross_validation_scores.mean()
    print(f"log_loss = {log_loss}")
    print(f"-log_loss = {-log_loss}")
    return -log_loss


def main():
    xgboostBO = BayesianOptimization(
        f=xgboostcv,
        pbounds=bounds_dict,
        random_state=None,
        verbose=2,
        bounds_transformer=None
    )
    print("maximize -logloss")
    xgboostBO.maximize(
        init_points=5,
        n_iter=50,
        acq='ucb',
        kappa=2.576,
        kappa_decay=1,
        kappa_decay_delay=0,
        xi=0.0,
        n_restarts_optimizer=50,
    )

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])

    # xgboostBO.maximize()

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])


def diffev():
    print("minimize logloss")
    result = differential_evolution(
        func=xgb_cost_function,
        bounds=bounds_list,
        # args=(),
        # strategy='best1bin',
        # maxiter=1000,
        # popsize=15,
        # tol=0.01,
        # mutation=(0.5, 1),
        # recombination=0.7,
        # seed=None,
        # callback=None,
        # disp=False,
        # polish=True,
        # init='latinhypercube',
        # atol=0,
        # updating='immediate',
        # workers=4,
    )
    print(result.x, result.fun)


if __name__ == "__main__":
    diffev()
