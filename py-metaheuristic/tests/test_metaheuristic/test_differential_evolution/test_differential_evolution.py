import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import differential_evolution
from py_metaheuristic.differential_evolution import de
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
    pprint(dir(differential_evolution))
    pprint(dir(de))


def test_differential_evolution(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    de_search = de.differential_evolution(
        target_function=easom,
        n=500,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
        f=0.9,
        cr=0.2,
    )

    variables = de_search[:-1]
    minimum = de_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]
