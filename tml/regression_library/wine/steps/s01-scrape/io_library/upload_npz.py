from io import BytesIO

import boto3
import numpy as np


def upload_npz(bucket, key, nda):
    buffer = BytesIO()
    s3_resource = boto3.resource('s3')
    np.savez(buffer, nda)
    obj = s3_resource.Object(bucket, key)
    obj.put(Body=buffer.getvalue())
