"""
Chapter 3. K-Nearest Neighbors (CRUD version)
"""

import random
import sys
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)


class DatasetManager:
    """
    CRUD operations for housing dataset.
    """
    def __init__(self):
        self.houses = None
        self.values = None

    # CREATE
    def create_from_csv(self, csv_file, limit=None):
        houses = pd.read_csv(csv_file, nrows=limit)
        self.values = houses['AppraisedValue']
        houses = houses.drop('AppraisedValue', axis=1)
        houses = (houses - houses.mean()) / (houses.max() - houses.min())
        self.houses = houses[['lat', 'long', 'SqFtLot']]

    # READ
    def read_sample(self, n=5):
        if self.houses is None:
            raise ValueError("Dataset not loaded.")
        return self.houses.head(n), self.values.head(n)

    # UPDATE
    def update_column(self, column_name, func):
        if column_name not in self.houses:
            raise KeyError(f"Column {column_name} not found.")
        self.houses[column_name] = self.houses[column_name].apply(func)

    # DELETE
    def delete_column(self, column_name):
        if column_name in self.houses:
            self.houses = self.houses.drop(column_name, axis=1)


class KNNModel:
    """
    CRUD operations for kNN regression model.
    """
    def __init__(self, k=5, metric=np.mean):
        self.k = k
        self.metric = metric
        self.kdtree = None
        self.houses = None
        self.values = None

    # CREATE
    def create_from_dataset(self, houses, values):
        self.houses = houses
        self.values = values
        self.kdtree = KDTree(self.houses)

    # READ
    def predict(self, query_point):
        if self.kdtree is None:
            raise ValueError("Model not trained.")
        _, indexes = self.kdtree.query(query_point, self.k)
        value = self.metric(self.values.iloc[indexes])
        if np.isnan(value):
            raise ValueError("Unexpected result")
        return value

    # UPDATE
    def update_k(self, new_k):
        self.k = new_k

    def update_metric(self, new_metric):
        self.metric = new_metric

    # DELETE
    def delete_model(self):
        self.kdtree = None
        self.houses = None
        self.values = None


class RegressionExperiment:
    """
    Run tests and experiments on dataset/model.
    """
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

    def tests(self, folds=5, k=5):
        holdout = 1 / float(folds)
        errors = []
        for _ in range(folds):
            values_pred, values_actual = self._test_regression(holdout, k)
            errors.append(mean_absolute_error(values_actual, values_pred))
        return errors

    def _test_regression(self, holdout, k):
        houses, values = self.dataset_manager.houses, self.dataset_manager.values
        test_rows = random.sample(houses.index.tolist(), int(round(len(houses) * holdout)))
        train_rows = set(range(len(houses))) - set(test_rows)

        df_train = houses.drop(test_rows)
        train_values = values.iloc[list(train_rows)]

        model = KNNModel(k=k)
        model.create_from_dataset(df_train, train_values)

        values_pred, values_actual = [], []
        for idx, row in houses.iloc[test_rows].iterrows():
            values_pred.append(model.predict(row))
            values_actual.append(values[idx])

        return values_pred, values_actual

    def plot_error_rates(self, folds_range=range(2, 11), k=5):
        errors_df = pd.DataFrame({'max': 0, 'min': 0}, index=folds_range)
        for folds in folds_range:
            errors = self.tests(folds=folds, k=k)
            errors_df.loc[folds, 'max'] = max(errors)
            errors_df.loc[folds, 'min'] = min(errors)
        errors_df.plot(title='Mean Absolute Error of KNN over different folds')
        plt.xlabel('#folds')
        plt.ylabel('MAE')
        plt.show()


def main():
    dataset = DatasetManager()
    dataset.create_from_csv('./data/king_county_data_geocoded.csv', limit=100)

    print("Sample data:")
    print(dataset.read_sample())

    experiment = RegressionExperiment(dataset)
    experiment.plot_error_rates()


if __name__ == '__main__':
    main()
