import json
from typing import Union

import boto3
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from io_library.write_dict import save_dict_to_json_fs, save_dict_to_json_s3


def convert_python_to_numpy(params_dict: dict) -> dict:
    """
    Convert a dictionary with base Python float and list types
    to a new dictionary with values as np.float64 and np.array(s).

    Parameters:
        params_dict (dict): Input dictionary with base Python types.

    Returns:
        dict: Output dictionary with values as np.float64 and np.array(s).
    """
    result = {}
    for key, value in params_dict.items():
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, float):
            value = np.float64(value)
        elif isinstance(value, bool):
            value = np.bool_(value)
        elif isinstance(value, int):
            value = np.int64(value)
        result[key] = value
    return result


def convert_numpy_to_python(params_dict: dict) -> dict:
    """
    Convert a dictionary with values as np.float64 and np.array(s)
    to a new dictionary with base Python float and list types.

    Parameters:
        params_dict (dict): Input dictionary with values as np.float64 and np.array(s).

    Returns:
        dict: Output dictionary with base Python float and list types.
    """
    result = {}
    for key, value in params_dict.items():
        print(f"key={key}", f"value={value}")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.number):
            value = float(value)
        if isinstance(value, np.bool_):
            value = bool(value)
        result[key] = value
    return result


def get_skl_to_dict(skl: TransformerMixin) -> dict:
    """
    Access the internal __dict__ representation of a scikit-learn object
    and convert datatypes suitable for saving to .json format.

    Parameters:
        skl (TransformerMixin): Scikit-learn object.

    Returns:
        dict: Dictionary with converted datatypes.
    """
    params_dict = convert_numpy_to_python(skl.__dict__)
    return params_dict


def get_dict_to_skl(params_dict: dict, skl: TransformerMixin):
    """
    Update the internal __dict__ of a scikit-learn object with numpy equivalent values.

    Parameters:
        params_dict (dict): Dictionary with numpy equivalent values.
        skl (TransformerMixin): Scikit-learn object to be updated.
    """
    params_dict_np = convert_python_to_numpy(params_dict)
    skl.__dict__.update(params_dict_np)


def save_skl_to_json_fs(skl: Union[TransformerMixin, BaseEstimator], filename: str):
    """
    Save a scikit-learn object to a local .json file.

    Parameters:
        skl (TransformerMixin): Scikit-learn object to be saved.
        filename (str): Local file path.
    """
    params_dict = get_skl_to_dict(skl)
    save_dict_to_json_fs(params_dict, filename)


def save_skl_to_json_s3(skl: TransformerMixin, bucket: str, key: str):
    """
    Save a scikit-learn object to a local .json file.

    Parameters:
        skl (TransformerMixin): Scikit-learn object to be saved.
        bucket (str): S3 bucket name.
        key (str): S3 key (file path).
    """
    params_dict = get_skl_to_dict(skl)
    save_dict_to_json_s3(params_dict, bucket, key)


def load_skl_from_json_fs(skl: TransformerMixin, filename: str):
    """
    Load a scikit-learn object from a local .json file.

    Parameters:
        skl (TransformerMixin): Scikit-learn object to be loaded.
        filename (str): Local file path.
    """
    params_bytes = open(filename, 'rb').read()
    params_str = params_bytes.decode('utf-8')
    params_dict = json.loads(params_str)
    get_dict_to_skl(params_dict, skl)


def load_skl_from_json_s3(skl: TransformerMixin, bucket: str, key: str):
    """
    Load a scikit-learn object from a .json file in an S3 bucket.

    Parameters:
        skl (TransformerMixin): Scikit-learn object to be loaded.
        bucket (str): S3 bucket name.
        key (str): S3 key (file path).
    """
    s3_resource = boto3.resource('s3')
    obj = s3_resource.Object(bucket, key)
    json_bytes = obj.get()['Body'].read()
    json_str = json_bytes.decode('utf-8')
    params = json.loads(json_str)
    get_dict_to_skl(params, skl)
