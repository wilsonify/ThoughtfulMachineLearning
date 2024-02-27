import json
import os.path

import pytest

from s01_wine_scrape.parse_one_event import parse_one_event, parse_first_event

path_to_here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(name="s3_event")
def s3_event_fixture():
    return json.load(open(f"{path_to_here}/s3-put.json", "r"))


@pytest.fixture(name="minimal_event")
def minimal_event_fixture():
    return json.load(open(f"{path_to_here}/example.json", "r"))


@pytest.fixture(name="body_event")
def body_event_fixture():
    return json.load(open(f"{path_to_here}/example2.json", "r"))


@pytest.fixture(name="sqs_event")
def sqs_event_fixture():
    return json.load(open(f"{path_to_here}/sqs.json", "r"))


def test_parse_sqs_event(sqs_event):
    result = parse_first_event(sqs_event)
    assert result == {
        'strategy': 'main_scrape_one_wine',
        'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'
    }


def test_parse_body_event(body_event):
    result = parse_first_event(body_event)
    assert result == {
        'strategy': 'main_scrape_one_wine',
        'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'
    }


def test_parse_first_event_minimal_event(minimal_event):
    result = parse_first_event(minimal_event)
    assert result == {
        'strategy': 'main_scrape_one_wine',
        'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'
    }


def test_parse_one_event_minimal_event(minimal_event):
    result = parse_one_event(minimal_event)
    assert result == {
        'strategy': 'main_scrape_one_wine',
        'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'
    }


def test_parse_s3_event(s3_event):
    result = parse_one_event(s3_event)
    assert result == {
        'Records': [
            {'awsRegion': 'us-east-1', 'eventName': 'ObjectCreated:Put', 'eventSource': 'aws:s3',
             'eventTime': '1970-01-01T00:00:00.000Z', 'eventVersion': '2.0',
             'requestParameters': {'sourceIPAddress': '127.0.0.1'},
             'responseElements': {'x-amz-id-2': 'EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH',
                                  'x-amz-request-id': 'EXAMPLE123456789'},
             's3': {'bucket': {'arn': 'arn:aws:s3:::example-bucket', 'name': 'example-bucket',
                               'ownerIdentity': {'principalId': 'EXAMPLE'}}, 'configurationId': 'testConfigRule',
                    'object': {'eTag': '0123456789abcdef0123456789abcdef', 'key': 'test%2Fkey',
                               'sequencer': '0A1B2C3D4E5F678901', 'size': 1024}, 's3SchemaVersion': '1.0'},
             'userIdentity': {'principalId': 'EXAMPLE'}}
        ]}
