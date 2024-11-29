from io import StringIO

import pandas as pd

from tml.c05_forests.decision_trees.classifier import MushroomForest
from tml.c05_forests.decision_trees.tests.mushroom_problem_test import CSV_CONTENT

problem = MushroomForest(StringIO(CSV_CONTENT))


def test_forest_training():
    """Test if the RandomForestClassifier trains without errors."""
    X = problem.data_frame[problem.features]
    Y = problem._MushroomProblem__factorize(problem.data_frame)
    classifier = problem.train(X, Y)
    assert classifier is not None


def test_forest_validation():
    """Test if confusion matrices are calculated."""
    folds = 2
    confusion_matrices = problem.validate(folds)
    assert len(confusion_matrices) == folds
    for cm in confusion_matrices:
        assert isinstance(cm, pd.DataFrame)
