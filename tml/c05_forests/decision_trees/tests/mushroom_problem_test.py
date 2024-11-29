from io import StringIO

import numpy as np
import pytest

from tml.c05_forests.decision_trees.classifier import MushroomProblem

# Sample CSV content for testing
CSV_CONTENT = """class,cap-shape,cap-surface,cap-color,bruises,odor
e,b,s,w,t,l
p,x,s,y,t,m
e,f,g,n,f,a
p,f,g,y,f,l
e,b,s,y,t,n
"""

data_file = StringIO(CSV_CONTENT)
problem = MushroomProblem(data_file)


def test_data_loading():
    """Test if the data is correctly loaded and factorized."""
    assert problem.data_frame.shape == (5, 6)
    assert 'class' in problem.data_frame.columns
    assert np.issubdtype(problem.data_frame['cap-shape'].dtype, np.integer)


def test_classes_extraction():
    """Test if classes are correctly extracted."""
    expected_classes = np.array(['e', 'p'])
    np.testing.assert_array_equal(problem.classes, expected_classes)


def test_invalid_folds():
    """Test if an error is raised for invalid number of folds."""
    with pytest.raises(AssertionError):
        problem.validation_data(folds=10)
