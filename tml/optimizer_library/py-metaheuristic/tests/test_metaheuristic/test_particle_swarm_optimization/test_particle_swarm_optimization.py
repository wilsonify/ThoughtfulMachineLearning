import math
from pprint import pprint

import pytest

from py_metaheuristic.particle_swarm_optimization.pso import particle_swarm_optimization, easom


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
    pprint(dir(particle_swarm_optimization))



def test_particle_swarm_optimization():
    """
    suppose that our Target Function is the Easom Function
    (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)
    """
    pso_search = particle_swarm_optimization(
        target_function=easom,
        swarm_size=250,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=500,
        decay=0,
        w=0.9,
        c1=2,
        c2=2,
    )

    variables = pso_search[:-1]
    minimum = pso_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]
