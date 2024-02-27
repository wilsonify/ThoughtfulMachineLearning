import json


def parse_one_event(message):
    body_dict = message
    if isinstance(message, str):
        body_dict = json.loads(message)

    parsed_event = {}
    for key, value in body_dict.items():
        if hasattr(value, 'items') and callable(getattr(value, 'items')):
            parsed_event[key] = dict(value)
        else:
            parsed_event[key] = value
    return parsed_event


def parse_first_event(event):
    messages_list = event.get('Records', [event])
    message = messages_list[0]
    body = message.get("body", event)

    if isinstance(body, str):
        body_dict = json.loads(body)
    elif isinstance(body, dict):
        body_dict = body

    parsed_event = {}
    for key, value in body_dict.items():
        if hasattr(value, 'items') and callable(getattr(value, 'items')):
            parsed_event[key] = dict(value)
        else:
            parsed_event[key] = value
    return parsed_event
