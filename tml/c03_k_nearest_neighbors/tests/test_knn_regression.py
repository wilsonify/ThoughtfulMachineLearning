import numpy as np
import pandas as pd
import pytest

from tml.c03_k_nearest_neighbors.regression import (
    DatasetManager,
    KNNModel,
    RegressionExperiment
)


@pytest.fixture
def sample_dataset():
    houses = pd.DataFrame({
        'lat': [47.5, 47.6, 47.7, 47.8, 47.9],
        'long': [-122.3, -122.2, -122.1, -122.0, -121.9],
        'SqFtLot': [5000, 6000, 7000, 8000, 9000]
    })
    values = pd.Series([300000, 400000, 500000, 600000, 700000])
    dm = DatasetManager()
    dm.houses = houses
    dm.values = values
    return dm


def test_dataset_read(sample_dataset):
    houses, values = sample_dataset.read_sample(2)
    assert len(houses) == 2
    assert len(values) == 2


def test_dataset_update(sample_dataset):
    sample_dataset.update_column('SqFtLot', lambda x: x + 100)
    assert all(sample_dataset.houses['SqFtLot'] == pd.Series([5100, 6100, 7100, 8100, 9100]))


def test_dataset_delete(sample_dataset):
    sample_dataset.delete_column('SqFtLot')
    assert 'SqFtLot' not in sample_dataset.houses.columns


def test_knnmodel_create_and_predict(sample_dataset):
    model = KNNModel(k=2)
    model.create_from_dataset(sample_dataset.houses, sample_dataset.values)
    query_point = sample_dataset.houses.iloc[0]
    prediction = model.predict(query_point)
    assert isinstance(prediction, (int, float, np.floating))


def test_knnmodel_update_k(sample_dataset):
    model = KNNModel(k=2)
    model.create_from_dataset(sample_dataset.houses, sample_dataset.values)
    model.update_k(3)
    assert model.k == 3


def test_knnmodel_delete_model(sample_dataset):
    model = KNNModel()
    model.create_from_dataset(sample_dataset.houses, sample_dataset.values)
    model.delete_model()
    assert model.kdtree is None


def test_regression_experiment_errors(sample_dataset):
    experiment = RegressionExperiment(sample_dataset)
    errors = experiment.tests(folds=3, k=2)
    assert isinstance(errors, list)
    assert all(isinstance(e, float) for e in errors)
    assert len(errors) == 3
