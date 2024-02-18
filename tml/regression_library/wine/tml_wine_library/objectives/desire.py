from functools import partial

from scipy.stats.mstats import gmean


def desire(h, low, target, high):
    if h <= low:
        return 0.0
    if h >= high:
        return 0.0
    if low < h <= target:
        return -low / (-low + target) + h / (-low + target)
    if target <= h < high:
        return 1.0 + target / (high - target) - h / (high - target)


def composite(pdes, wd):
    return gmean([pdes, wd], axis=0)


pdesire = partial(desire, low=0, target=10, high=110)

wdesire = partial(desire, low=85, target=99, high=100)
