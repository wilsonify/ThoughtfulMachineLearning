import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import dispersive_flies_optimization
from py_metaheuristic.dispersive_flies_optimization import dfo
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
    pprint(dir(dispersive_flies_optimization))
    pprint(dir(dfo))


def test_dispersive_flies_optimization(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    dfo_search = dfo.dispersive_fly_optimization(
        target_function=easom,
        swarm_size=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        generations=100,
        thresh=0.2
    )

    variables = dfo_search[:-1]
    minimum = dfo_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.0001)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.001),
        pytest.approx(math.pi, abs=0.001),
    ]
