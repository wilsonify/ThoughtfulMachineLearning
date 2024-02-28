import re
from typing import Union

import numpy as np
import pandas as pd


def extract_number(input_string):
    pattern = r'(\d+)ML'
    try:
        match = re.search(pattern, input_string)
        match_group = match.group(1)
        number = float(match_group)
    except:
        number = np.nan
    return number


def format_size(input_data: Union[pd.Series, np.ndarray, list], column_name: str) -> pd.Series:
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        # Convert input data to pandas DataFrame with specified column name
        df = pd.DataFrame({column_name: input_data})

    # Apply the extract_number function to the specified column
    df[column_name] = df[column_name].astype(str).apply(extract_number)
    return df[column_name]


def test_list_usage():
    # Applying the function to the series
    input_list = ["750ML", "500ML", "1000ML"]
    result = format_size(input_list, "Size")
    assert list(result) == [750, 500, 1000]


def test_series_usage():
    # Applying the function to the series
    input_series = pd.Series(["750ML", "500ML", "1000ML"])
    result = format_size(input_series, "Size")
    assert list(result) == [750, 500, 1000]


def test_nda_usage():
    # Applying the function to the numpy array
    input_array = np.array(["750ML", "500ML", "1000ML"])
    result = format_size(input_array, "Size")
    assert list(result) == [750, 500, 1000]


def test_df_usage():
    input_df = pd.DataFrame({"Size": ["750ML", "500ML", "1000ML"], "Price": [10, 20, 30]})
    result = format_size(input_df, "Size")
    assert list(result) == [750, 500, 1000]
