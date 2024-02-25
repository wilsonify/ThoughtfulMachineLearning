from fnmatch import fnmatch

from boto3 import Session


def list_objects_s3(bucket, prefix, glob_pattern='*'):
    s3_session = Session()
    s3_client = s3_session.client('s3')
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    # Extract file names from the response that match the glob pattern
    files_list = [obj['Key'] for obj in response.get('Contents', []) if fnmatch(obj['Key'], glob_pattern)]
    return files_list


def list_directories_s3(bucket, prefix):
    """
    List directories in the first level of an S3 prefix.

    :param bucket: The S3 bucket name.
    :param prefix: The S3 object prefix.
    :return: List of directories.
    """
    s3_session = Session()
    s3_client = s3_session.client('s3')

    # List objects in the specified S3 prefix
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)

    # Extract directory names from the response
    directories_list = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('/')]

    return directories_list
