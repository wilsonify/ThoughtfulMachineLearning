import glob
import json
import logging
import os

import boto3

from mlflow_tf.io_library.ConfigDataClass import Config
from mlflow_tf.io_library.write_dict import save_dict_to_json_s3


def log_artifact(local_file_path, bucket, prefix):
    """ log an arbitrary local file for this run """

    _, local_file_tail = os.path.split(local_file_path)
    key = f"{prefix}/{local_file_tail}"
    session = boto3.Session()
    s3 = session.client('s3')

    logging.info("%r", f"start uploading {local_file_path} to s3://{bucket}/{key}")
    s3.upload_file(local_file_path, bucket, key)
    logging.info("%r", f"done uploading {local_file_path} to s3://{bucket}/{key}")


def log_params(configuration: Config, bucket, prefix):
    """ save the configuration that was used for training """
    configuration.to_json_s3(
        bucket=bucket,
        key=f"{prefix}/config.json"
    )


def log_model(model_obj, signature_dict, bucket, prefix, saved_model_local_dir=None):
    """ save the model artifacts created during training """

    _, prefix_tail = os.path.split(prefix)

    if saved_model_local_dir is None:
        saved_model_local_dir = f"runs/{prefix_tail}"

    saved_model_local_path = f"{saved_model_local_dir}/saved_model"

    signature_key = f"{prefix}/signature.json"
    signature_str = json.dumps(signature_dict)
    signature_bytes = signature_str.encode("utf-8")
    model_obj.save(saved_model_local_path)
    session = boto3.Session()
    s3 = session.client('s3')
    s3.put_object(Body=signature_bytes, Bucket=bucket, Key=signature_key)
    logging.info("%r", f"Signature saved to S3: s3://{bucket}/{signature_key}")
    for checkpoint_file in glob.glob(f"{saved_model_local_path}/**/*", recursive=True):
        logging.debug("%r", f"checkpoint_file = {checkpoint_file}")
        try:
            log_artifact(
                local_file_path=checkpoint_file,
                bucket=bucket,
                prefix=f"{prefix}/saved_model"
            )
        except IsADirectoryError:
            logging.warning("%r", f"{checkpoint_file} is a directory, skip to next file")
            continue


def log_metrics(metrics_dict, bucket, key):
    """ save the configuration that was used for training """
    save_dict_to_json_s3(metrics_dict, bucket, key)
