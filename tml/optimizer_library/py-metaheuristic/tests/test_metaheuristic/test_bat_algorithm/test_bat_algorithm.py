import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import bat_algorithm
from py_metaheuristic.bat_algorithm import bat_a
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
    pprint(dir(bat_algorithm))
    pprint(dir(bat_a))


def test_bat_algorithm(front):
    """
    test_bat_algorithm
    :param front:
    :return:
    """
    # BA - Algorithm
    bat_search = bat_a.bat_algorithm(
        target_function=easom,
        swarm_size=7500,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=5,
        alpha=0.8,
        gama=0.8,
        fmin=0,
        fmax=2,
    )

    variables = bat_search[:-1]
    minimum = bat_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]
