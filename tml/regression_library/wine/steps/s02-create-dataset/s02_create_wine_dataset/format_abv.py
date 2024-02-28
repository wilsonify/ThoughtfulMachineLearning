import re
from typing import Union

import numpy as np
import pandas as pd


def extract_ABV(input_string: str) -> float:
    # Alchohol by Volume e.g. 14.5%
    pattern = r'(\d+\.\d+|\d+)%'
    try:
        # Use re.search to find the first occurrence of the pattern in the string
        match = re.search(pattern, input_string)
        match_group = match.group(1)
        number = float(match_group)
    except:
        number = np.nan
    return number


def format_ABV(input_data: Union[pd.Series, np.ndarray, list], column_name: str) -> pd.Series:
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        # Convert input data to pandas DataFrame with specified column name
        df = pd.DataFrame({column_name: input_data})

    # Apply the extract_number function to the specified column
    df[column_name] = df[column_name].astype(str).apply(extract_ABV)
    return df[column_name]


def test_list_usage():
    # Applying the function to the series
    input_list = ["7.5%", "5.00%", "10%"]
    result = format_ABV(input_list, "Size")
    assert list(result) == [7.5, 5.0, 10.0]


def test_series_usage():
    # Applying the function to the series
    input_series = pd.Series(["7.5%", "5.00%", "10%"])
    result = format_ABV(input_series, "Size")
    assert list(result) == [7.5, 5.0, 10.0]


def test_nda_usage():
    # Applying the function to the numpy array
    input_array = np.array(["7.5%", "5.00%", "10%"])
    result = format_ABV(input_array, "Size")
    assert list(result) == [7.5, 5.0, 10.0]


def test_df_usage():
    input_df = pd.DataFrame({"Size": ["7.5%", "5.00%", "10%"], "Price": [10, 20, 30]})
    result = format_ABV(input_df["Size"], "Size")
    assert list(result) == [7.5, 5.0, 10.0]
