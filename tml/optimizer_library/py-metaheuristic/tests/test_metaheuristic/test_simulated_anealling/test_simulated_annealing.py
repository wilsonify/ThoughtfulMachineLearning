import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import simulated_anealling

from py_metaheuristic.simulated_anealling import sa
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
    pprint(dir(simulated_anealling))
    pprint(dir(sa))


def test_simulated_annealing(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    sa_search = sa.simulated_annealing(
        target_function=easom,
        min_values=[-5, -5],
        max_values=[5, 5],
        mu=0,
        sigma=1,
        initial_temperature=1.0,
        temperature_iterations=1000,
        final_temperature=0.0001,
        alpha=0.9,
    )

    variables = sa_search[0][:-1]
    minimum = sa_search[0][-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]
