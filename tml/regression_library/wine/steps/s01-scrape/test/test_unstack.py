import csv
from collections import defaultdict

import pandas as pd


def test_init():
    df = pd.read_csv("stacked.csv")
    df = df.drop_duplicates(subset=["time", "tag", "value"], keep='first')
    df = df.drop_duplicates(subset=["time", "tag"], keep='first')
    df = df.set_index(["time", "tag"])
    df = df.unstack("tag")
    df.columns = [_[1] for _ in df.columns]
    df = df.reset_index()
    df.index.name = "index"
    assert df.to_dict(orient="list") == {
        'tag1': [0.959506669577661, 0.503873011480695, 0.240073337683448,
                 0.667804926922008, 0.39186872598277, 0.264173942045898,
                 0.441097466215352, 0.615991093842308, 0.378933217602024],
        'tag2': [0.13402323997491, 0.780564570187529, 0.922842976084817,
                 0.922048005606614, 0.743041675501411, 0.829404333958447,
                 0.326510896213021, 0.499141487820921, 0.0852967818714977],
        'tag3': [0.725336573873206, 0.231297457798543, 0.500694758562708,
                 0.2431451889339, 0.0462510113871852, 0.0819302152329273,
                 0.58111532537678, 0.140172394434153, 0.964005941484954],
        'time': ['2024/03/02 07:53:00', '2024/03/02 07:54:00', '2024/03/02 07:55:00',
                 '2024/03/02 07:56:00', '2024/03/02 07:57:00', '2024/03/02 07:58:00',
                 '2024/03/02 07:59:00', '2024/03/02 08:00:00', '2024/03/02 08:01:00']}
    df.to_csv("flat.csv")


def transform_input_to_output(reader):
    all_tags = set([])
    all_times = set([])
    data = defaultdict(dict)
    for row in reader:
        time = row['time']
        tag = row['tag']
        value = row['value']
        all_tags.add(tag)
        all_times.add(time)
        if tag not in data[time]:
            data[time][tag] = [value]
        else:
            data[time][tag].append(value)

    result = {'time': sorted(all_times)}
    for tag in all_tags:
        result[tag] = []
        for time in result['time']:
            if time in data and tag in data[time]:
                result[tag].append(float(data[time][tag][0]))  # Take the first value
                data[time][tag] = data[time][tag][1:]  # Remove the taken value
            else:
                result[tag].append(None)  # Fill missing values with NaN

    return result


def test_unstack():
    with open('stacked.csv', newline='\n') as input_csvfile:
        reader = csv.DictReader(input_csvfile)
        output = transform_input_to_output(reader)
    assert output == {
        'tag1': [0.959506669577661, 0.503873011480695, 0.240073337683448, 0.667804926922008, 0.39186872598277,
                 0.264173942045898, 0.441097466215352, 0.615991093842308, 0.378933217602024],
        'tag2': [0.13402323997491, 0.780564570187529, 0.922842976084817, 0.922048005606614, 0.743041675501411,
                 0.829404333958447, 0.326510896213021, 0.499141487820921, 0.0852967818714977],
        'tag3': [0.725336573873206, 0.231297457798543, 0.500694758562708, 0.2431451889339, 0.0462510113871852,
                 0.0819302152329273, 0.58111532537678, 0.140172394434153, 0.964005941484954],
        'time': ['2024/03/02 07:53:00', '2024/03/02 07:54:00', '2024/03/02 07:55:00',
                 '2024/03/02 07:56:00', '2024/03/02 07:57:00', '2024/03/02 07:58:00',
                 '2024/03/02 07:59:00', '2024/03/02 08:00:00', '2024/03/02 08:01:00']}
