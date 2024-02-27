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


def lambda_handler(event, context):
    print("lambda_handler")
    assert 'Records' in event, "event from sqs must contain Records"
    for record in event['Records']:
        assert 'body' in record, "each record from sqs must contain body"
        body = record['body']
        message = json.loads(body)
        print(f"message = {message}")
        process_one_message(message)
        print("Finished processing one message")
    print("Finished processing all messages")
    response = {
        "statusCode": 200,
        "body": "Finished processing from s01_wine_scrape lambda"
    }
    return response
