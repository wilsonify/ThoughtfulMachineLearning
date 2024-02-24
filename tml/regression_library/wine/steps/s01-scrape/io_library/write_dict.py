import json

import boto3


def save_dict_to_json_s3(params_dict: dict, bucket: str, key: str):
    """
    Save a dictionary to a .json file in an S3 bucket.

    Parameters:
        params_dict (dict): Dictionary to be saved.
        bucket (str): S3 bucket name.
        key (str): S3 key (file path).
    """
    params_str = json.dumps(params_dict)
    params_bytes = params_str.encode('utf-8')
    s3_resource = boto3.resource('s3')
    obj = s3_resource.Object(bucket, key)
    obj.put(Body=params_bytes)


def save_dict_to_json_fs(params_dict: dict, filename: str):
    """
    Save a dictionary to a local .json file.

    Parameters:
        params_dict (dict): Dictionary to be saved.
        filename (str): Local file path.
    """
    params_str = json.dumps(params_dict)
    params_bytes = params_str.encode('utf-8')
    with open(filename, 'wb') as file:
        file.write(params_bytes)
