import inspect
import json
import logging
import time
import traceback
from logging.config import dictConfig

import boto3

from s01_wine_scrape.available import available

sqs = boto3.client('sqs')


def on_error(msg):
    logging.info("on_error")

    sqs.send_message(
        QueueUrl='wine-sqs-fail',
        MessageBody=json.dumps(msg)
    )


def on_success(msg):
    logging.info("on_success")
    assert "strategy" in msg, "message must contain strategy"
    strat_str = msg["strategy"]
    strat_func = available[strat_str]
    strat_func_sig = inspect.signature(strat_func)
    valid_keys = strat_func_sig.parameters.keys()
    valid_values = {k: msg[k] for k in valid_keys}
    logging.info(f"start {strat_str}")
    strat_func(**valid_values)
    logging.info(f"done {strat_str}")

    msg_str = json.dumps(msg)
    sqs.send_message(
        QueueUrl='wine-sqs-done',
        MessageBody=msg_str
    )


def process_one_message(message):
    try:
        logging.info("happy path")
        on_success(message)
        logging.info("success")
    except Exception as e:
        logging.info(f"Error: {e}")
        logging.info("unhappy path")
        try:
            message['error_stack_trace'] = traceback.format_exc()  # Add stack trace to the event
        except:
            logging.info("could not capture stack trace")
        on_error(message)


def parse_event(event):
    logging.info("lambda_handler")
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
    logging.info("Start processing all messages")
    for record in event_parsed['Records']:
        message = parse_record(record)
        logging.info(f"Start processing one message = {message}")
        process_one_message(message)
        logging.info("Done processing one message")


def lambda_handler(event, context):
    logging.info("lambda_handler")
    logging.info("start processing all records in series")
    process_records_in_series(event)
    logging.info("done processing all records in series")
    response = {
        "statusCode": 200,
        "body": "Finished processing from s01_wine_scrape lambda"
    }
    return response


def process_batch_in_series(batch):
    logging.info("start process a batch of messages in series")
    assert 'Messages' in batch, "batch must contain Messages"
    messages = batch['Messages']
    n_messages = len(messages)
    for count, message in enumerate(messages):
        logging.info(f"{count}/{n_messages}")
        receipt_handle = message['ReceiptHandle']
        try:
            process_records_in_series(message['Body'])
        except Exception as err:
            logging.info(f"cool off to prevent tight loop on Error:{err}")
            time.sleep(5)
        logging.info("Acknowledge message")
        sqs.delete_message(
            QueueUrl='wine-sqs-try',
            ReceiptHandle=receipt_handle
        )


if __name__ == "__main__":
    dictConfig(dict(
        version=1,
        formatters={"simple": {"format": """%(asctime)s | %(name)s | %(lineno)s | %(levelname)s | %(message)s"""}},
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.DEBUG},
    ))
    while True:
        logging.info("Receive next batch of messages from the queue")
        process_batch_in_series(sqs.receive_message(
            QueueUrl='wine-sqs-try',
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20  # Long polling to reduce cost and improve responsiveness
        ))
        logging.info("Wait for next batch of messages from the queue")
        time.sleep(10)
