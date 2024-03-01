import inspect
import json
import traceback

import boto3

from s01_wine_scrape.available import available


def on_error(msg):
    print("on_error")
    sqs = boto3.client('sqs')
    sqs.send_message(
        QueueUrl='wine-sqs-fail',
        MessageBody=json.dumps(msg)
    )


def on_success(msg):
    print("on_success")
    assert "strategy" in msg, "message must contain strategy"
    strat_str = msg["strategy"]
    strat_func = available[strat_str]
    strat_func_sig = inspect.signature(strat_func)
    valid_keys = strat_func_sig.parameters.keys()
    valid_values = {k: msg[k] for k in valid_keys}
    print(f"start {strat_str}")
    strat_func(**valid_values)
    print(f"done {strat_str}")
    sqs = boto3.client('sqs')
    msg_str = json.dumps(msg)
    sqs.send_message(
        QueueUrl='wine-sqs-done',
        MessageBody=msg_str
    )


def process_one_message(message):
    try:
        print("happy path")
        on_success(message)
        print("success")
    except Exception as e:
        print(f"Error: {e}")
        print("unhappy path")
        try:
            message['error_stack_trace'] = traceback.format_exc()  # Add stack trace to the event
        except:
            print("could not capture stack trace")
        on_error(message)


def parse_event(event):
    print("lambda_handler")
    if 'Records' not in event:
        event = {"Records": [event]}
    assert 'Records' in event, "event from sqs must contain Records"
    return event


def parse_record(record):
    if 'body' not in record:
        record = {"body": record}
    assert 'body' in record, "each record from sqs must contain body"
    body = record['body']
    message = body
    if isinstance(body, (str, bytes, bytearray)):
        message = json.loads(body)
    return message


def process_records_in_series(event):
    event_parsed = parse_event(event)
    for record in event['Records']:
        message = parse_record(record)
        print(f"Start processing one message = {message}")
        process_one_message(message)
        print("Done processing one message")


def lambda_handler(event, context):
    print("lambda_handler")
    process_records_in_series(event)
    print("Finished processing all messages")
    response = {
        "statusCode": 200,
        "body": "Finished processing from s01_wine_scrape lambda"
    }
    return response
