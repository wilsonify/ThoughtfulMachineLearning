import json
import os.path

import pytest

from s01_wine_scrape.__main__ import parse_record, parse_event
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


def test_parse_event_s3(s3_event):
    result = parse_event(s3_event)
    assert result == {'Records': [{
        'awsRegion': 'us-east-1',
        'eventName': 'ObjectCreated:Put',
        'eventSource': 'aws:s3',
        'eventTime': '1970-01-01T00:00:00.000Z',
        'eventVersion': '2.0',
        'requestParameters': {'sourceIPAddress': '127.0.0.1'},
        'responseElements': {
            'x-amz-id-2': 'EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH',
            'x-amz-request-id': 'EXAMPLE123456789'},
        's3': {'bucket': {'arn': 'arn:aws:s3:::example-bucket',
                          'name': 'example-bucket',
                          'ownerIdentity': {'principalId': 'EXAMPLE'}},
               'configurationId': 'testConfigRule',
               'object': {'eTag': '0123456789abcdef0123456789abcdef',
                          'key': 'test%2Fkey',
                          'sequencer': '0A1B2C3D4E5F678901',
                          'size': 1024},
               's3SchemaVersion': '1.0'},
        'userIdentity': {'principalId': 'EXAMPLE'}}]}


def test_parse_event_body(body_event):
    result = parse_event(body_event)
    assert result == {'Records': [{'body': {
        'strategy': 'main_scrape_one_wine',
        'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'}}]}


def test_parse_event_sqs(sqs_event):
    result = parse_event(sqs_event)
    assert result == {'Records': [{
        'attributes': {'ApproximateFirstReceiveTimestamp': '1523232000001',
                       'ApproximateReceiveCount': '1',
                       'SenderId': '123456789012',
                       'SentTimestamp': '1523232000000'},
        'awsRegion': 'us-east-1',
        'body': '{\n'
                '    "strategy": "main_scrape_one_wine",\n'
                '    "wine_key": '
                '"wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html"\n'
                '  }',
        'eventSource': 'aws:sqs',
        'eventSourceARN': 'arn:aws:sqs:us-east-1:123456789012:MyQueue',
        'md5OfBody': '{{{md5_of_body}}}',
        'messageAttributes': {},
        'messageId': '19dd0b57-b21e-4ac1-bd88-01bbb068cb78',
        'receiptHandle': 'MessageReceiptHandle'}]}


def test_parse_record_body(body_event):
    body_event = parse_event(body_event)
    result = parse_record(body_event)
    assert result == {
        'Records': [{'body': {'strategy': 'main_scrape_one_wine',
                              'wine_key': 'wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html'}}]}


def test_parse_record_s3(s3_event):
    s3_event = parse_event(s3_event)
    result = parse_record(s3_event)
    assert result == {
        'Records': [
            {'awsRegion': 'us-east-1',
             'eventName': 'ObjectCreated:Put',
             'eventSource': 'aws:s3',
             'eventTime': '1970-01-01T00:00:00.000Z',
             'eventVersion': '2.0',
             'requestParameters': {'sourceIPAddress': '127.0.0.1'},
             'responseElements': {
                 'x-amz-id-2': 'EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH',
                 'x-amz-request-id': 'EXAMPLE123456789'},
             's3': {'bucket': {'arn': 'arn:aws:s3:::example-bucket',
                               'name': 'example-bucket',
                               'ownerIdentity': {'principalId': 'EXAMPLE'}},
                    'configurationId': 'testConfigRule',
                    'object': {'eTag': '0123456789abcdef0123456789abcdef',
                               'key': 'test%2Fkey',
                               'sequencer': '0A1B2C3D4E5F678901',
                               'size': 1024},
                    's3SchemaVersion': '1.0'},
             'userIdentity': {'principalId': 'EXAMPLE'}}]}


def test_parse_record_sqs(sqs_event):
    sqs_event = parse_event(sqs_event)
    result = parse_record(sqs_event)
    assert result == {
        'Records': [
            {'attributes': {'ApproximateFirstReceiveTimestamp': '1523232000001', 'ApproximateReceiveCount': '1',
                            'SenderId': '123456789012', 'SentTimestamp': '1523232000000'},
             'awsRegion': 'us-east-1', 'body':
                 '{\n'
                 '    "strategy": "main_scrape_one_wine",\n'
                 '    "wine_key": '
                 '"wine/s01-scrape/2024-02-25/html/wines/product_10000-hours-cabernet-sauvignon-2019_834207.html"\n'
                 '  }',
             'eventSource': 'aws:sqs',
             'eventSourceARN': 'arn:aws:sqs:us-east-1:123456789012:MyQueue',
             'md5OfBody': '{{{md5_of_body}}}',
             'messageAttributes': {},
             'messageId': '19dd0b57-b21e-4ac1-bd88-01bbb068cb78',
             'receiptHandle': 'MessageReceiptHandle'}]}


def test_process_records_in_series_s3(s3_event):
    event = s3_event
    event = parse_event(event)
    for record in event['Records']:
        message = parse_record(record)
        print(message)


def test_process_records_in_series_body(body_event):
    event = body_event
    event = parse_event(event)
    for record in event['Records']:
        message = parse_record(record)
        print(message)


def test_process_records_in_series_sqs(sqs_event):
    event = sqs_event
    event = parse_event(event)
    for record in event['Records']:
        message = parse_record(record)
        print(message)
