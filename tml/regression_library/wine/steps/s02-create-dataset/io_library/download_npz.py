import logging
from io import BytesIO

import boto3
import numpy as np


def download_npz(bucket, key):
    session = boto3.Session()
    s3_client = session.client('s3')
    credential_method = session.get_credentials().method
    logging.debug("%r", f"credential_method = {credential_method}")
    logging.debug("%r", f"bucket/key = {bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response['Body']
    body_bytes = body.read()
    buffer = BytesIO(body_bytes)
    npz = np.load(buffer)
    first_item = next(iter(npz.files))
    logging.debug("%r", f"first_item = {first_item}")
    nda = npz[first_item]
    logging.debug("%r", f"nda.shape = {nda.shape}")
    return nda
