import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import cross_entropy_method
from py_metaheuristic.cross_entropy_method import cem
import math
def easom(variables_values=(0, 0)):
    """
    Target Function: Easom Function
    :param variables_values:
    :return:
    """
    return (
            -math.cos(variables_values[0]) * math.cos(variables_values[1]) *
            math.exp(-((variables_values[0] - math.pi) ** 2) - (variables_values[1] - math.pi) ** 2))



def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(py_metaheuristic))
    pprint(dir(cross_entropy_method))
    pprint(dir(cem))


def test_cross_entropy_method(front):
    """
    test_cross_entropy_method
    :param front:
    :return:
    """

    cem_search = cem.cross_entropy_method(
        target_function=easom,
        n=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
        learning_rate=0.7,
        k_samples=15,
    )
    variables = cem_search[:-1]
    minimum = cem_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.05),
        pytest.approx(math.pi, abs=0.05),
    ]
