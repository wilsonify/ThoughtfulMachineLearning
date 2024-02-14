import math
from pprint import pprint

import py_metaheuristic
import pytest
from py_metaheuristic import cuckoo_search
from py_metaheuristic.cuckoo_search import cuckoo_s
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
    pprint(dir(cuckoo_search))
    pprint(dir(cuckoo_s))


def test_cuckoo_search(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    cs = cuckoo_s.cuckoo_search(
        target_function=easom,
        birds=500,
        discovery_rate=0.25,
        alpha_value=0.01,
        lambda_value=1.5,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=250,
    )
    variables = cs[:-1]
    minimum = cs[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]
