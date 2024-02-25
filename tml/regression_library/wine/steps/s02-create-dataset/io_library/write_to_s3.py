from io import BytesIO

from boto3 import Session


def write_parquet_to_s3(df, bucket, key):
    s3_session = Session()
    s3_client = s3_session.client('s3')
    buffer = BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )


def write_csv_to_s3(df, bucket, key):
    s3_session = Session()
    s3_client = s3_session.client('s3')
    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )
