source_abs_path="/home/thom/repos/serverless-architectures-aws/p01-first-steps/c02-architectures/twentyfour-hour-video/test/SampleVideo_1280x720_5mb.mp4"
dest_uri="s3://064592191516-serverless-video-upload-python"
aws --profile 064592191516 s3 cp $source_abs_path $dest_uri
