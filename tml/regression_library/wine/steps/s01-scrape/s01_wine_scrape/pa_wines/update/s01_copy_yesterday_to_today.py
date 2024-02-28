from datetime import datetime, timedelta

import boto3

from io_library.list_objects_s3 import list_objects_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def copy_s3_objects(
        source_bucket,
        source_prefix,
        destination_bucket,
        destination_prefix
):
    s3_client = boto3.client('s3')
    objects_to_copy = list_objects_s3(bucket=source_bucket, prefix=source_prefix)
    for obj in objects_to_copy:
        copy_source = {'Bucket': source_bucket, 'Key': obj}
        new_key = obj.replace(source_prefix, destination_prefix, 1)
        s3_client.copy_object(Bucket=destination_bucket, Key=new_key, CopySource=copy_source)
        print(f"copied {source_bucket}/{obj} to {destination_bucket}/{new_key}")


if __name__ == "__main__":
    today_date = datetime.now()
    yesterday_date = today_date - timedelta(days=1)
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')
    copy_s3_objects(
        source_bucket=OUTPUT_BUCKET,
        source_prefix=f'{OUTPUT_PREFIX}/{yesterday_date_str}/html/',
        destination_bucket=OUTPUT_BUCKET,
        destination_prefix=f'{OUTPUT_PREFIX}/{today_date_str}/html/'
    )
    copy_s3_objects(
        source_bucket=OUTPUT_BUCKET,
        source_prefix=f'{OUTPUT_PREFIX}/{yesterday_date_str}/json/',
        destination_bucket=OUTPUT_BUCKET,
        destination_prefix=f'{OUTPUT_PREFIX}/{today_date_str}/json/'
    )
