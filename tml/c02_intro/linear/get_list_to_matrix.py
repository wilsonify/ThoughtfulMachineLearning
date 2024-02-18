import numpy as np


def get_list_to_matrix(x_list):
    assert isinstance(x_list, list)
    x_array = np.array(x_list)
    result = x_array
    if x_array.ndim == 0:  # Scalar value
        result = x_array.reshape(1, 1)
    if x_array.ndim == 1:  # 1D list
        result = x_array.reshape(-1, 1)
    return result
