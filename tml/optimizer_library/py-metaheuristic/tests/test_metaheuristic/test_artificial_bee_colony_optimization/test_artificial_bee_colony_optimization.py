import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import artificial_bee_colony_optimization
from py_metaheuristic.artificial_bee_colony_optimization import abco
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
    pprint(dir(artificial_bee_colony_optimization))
    pprint(dir(abco))


def test_artificial_bee_colony_optimization(front):
    """
    test_artificial_bee_colony_optimization
    :param front:
    :return:
    """

    abco_instance = abco.ArtificialBeeColony(
        target_function=easom,
        food_sources=20,
        iterations=100,
        min_values=[-5, -5],
        max_values=[5, 5],
        employed_bees=20,
        outlookers_bees=20,
        limit=40
    )
    abco_search = abco_instance.minimize()

    variables = abco_search[:-1]
    minimum = abco_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.0001)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.001),
        pytest.approx(math.pi, abs=0.001),
    ]
