from io import StringIO

from tml.c05_forests.decision_trees.classifier import MushroomRegression
from tml.c05_forests.decision_trees.tests.mushroom_problem_test import CSV_CONTENT

problem = MushroomRegression(StringIO(CSV_CONTENT))


def test_regressor_training():
    """Test if the DecisionTreeRegressor trains without errors."""
    X = problem.data_frame[problem.features]
    Y = problem._MushroomProblem__factorize(problem.data_frame)
    regressor = problem.train(X, Y)
    assert regressor is not None


def test_regressor_validation():
    """Test if validation computes MSE correctly."""
    folds = 2
    mse_scores = problem.validate(folds)
    assert len(mse_scores) == folds
    assert all(score >= 0 for score in mse_scores)
