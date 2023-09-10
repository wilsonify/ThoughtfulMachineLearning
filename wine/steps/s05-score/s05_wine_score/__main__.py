import json
import os
from urllib.parse import unquote

import boto3

sqs = boto3.client('sqs')


def get_key_from_s3_put(event):
    key = event['Records'][0]['s3']['object']['key']
    source_key = unquote(key.replace('+', ' '))
    return source_key


def get_bucket_from_s3_put(event):
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    return bucket_name


audio_selector = dict(SelectorType='TRACK', Tracks=[1])
output1 = dict(
    Preset='System-Generic_Hd_Mp4_Avc_Aac_16x9_1920x1080p_24Hz_6Mbps',
    Extension='mp4',
    NameModifier='_16x9_1920x1080p_24Hz_6Mbps'
)
output2 = dict(
    Preset='System-Generic_Hd_Mp4_Avc_Aac_16x9_1280x720p_24Hz_4.5Mbps',
    Extension='mp4',
    NameModifier='_16x9_1280x720p_24Hz_4.5Mbps'
)
output3 = dict(
    Preset='System-Generic_Sd_Mp4_Avc_Aac_4x3_640x480p_24Hz_1.5Mbps',
    Extension='mp4',
    NameModifier='_4x3_640x480p_24Hz_1.5Mbps'
)


def happy_path(event):
    bucket_name = get_bucket_from_s3_put(event)
    source_key = get_key_from_s3_put(event)
    input_uri = f's3://{bucket_name}/{source_key}'
    output_bucket_name = os.getenv('TRANSCODED_VIDEO_BUCKET', "default-bucket")
    output_prefix = source_key.split('.')[0]
    output_uri = f's3://{output_bucket_name}/{output_prefix}/'
    role = os.getenv('MEDIA_ROLE', "default-media-role")
    mc_set_inputs = [dict(
        FileInput=input_uri,
        AudioSelectors={'Audio Selector 1': audio_selector}
    )]
    file_group_settings = dict(Destination=output_uri)
    output_group_settings = dict(
        Type='FILE_GROUP_SETTINGS',
        FileGroupSettings=file_group_settings
    )
    mc_set_output_groups = [dict(
        Name='File Group',
        Outputs=[output1, output2, output3],
        OutputGroupSettings=output_group_settings
    )]
    settings = dict(Inputs=mc_set_inputs, OutputGroups=mc_set_output_groups)
    job = dict(Role=role, Settings=settings)
    return job


def send_message(job_dict):
    print("Send message to SQS queue")
    default_queue_url = "https://sqs.us-east-1.amazonaws.com/064592191516/serverless-video-transcode-sqs-try"
    queue_url = os.getenv('SQS_QUEUE_URL', default_queue_url)
    job_str = json.dumps(job_dict)
    response = sqs.send_message(
        QueueUrl=queue_url,
        DelaySeconds=0,
        MessageBody=job_str
    )
    print(response['MessageId'])


def lambda_handler(event, context):
    media_convert_result = happy_path(event)
    print(media_convert_result)
    send_message(media_convert_result)
    return {
        'statusCode': 200,
        'body': json.dumps('Video transcoding job submitted successfully!')
    }
