resource "aws_lambda_function" "twentyfour-hour-video-python-transcode-video" {
  function_name                  = "twentyfour-hour-video-python-transcode-video"
  architectures                  = ["x86_64"]
  image_uri                      = "064592191516.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::064592191516:role/transcode-video"
  source_code_hash               = "34e2a1603ba3bceb7f0e51ccdef6d5a1c90c4d68af2bc3bf6af4491a13e8d688"
  environment {
    variables = {
      MEDIA_ENDPOINT          = "https://lxlxpswfb.mediaconvert.us-east-1.amazonaws.com"
      MEDIA_ROLE              = "arn:aws:iam::064592191516:role/media-convert-role"
      TRANSCODED_VIDEO_BUCKET = "064592191516-serverless-video-transcode-python"
    }
  }
  image_config { command = ["transcode_video_python/__main__.lambda_handler"] }
  tags = {
    STAGE = "dev"
  }
  timeout = "6"
  tracing_config {
    mode = "PassThrough"
  }

}

