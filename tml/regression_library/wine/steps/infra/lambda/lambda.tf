variable "aws_account_number" {
  description = "AWS Account Number"
  type        = string
  default     = "064592191516"
}

resource "aws_lambda_function" "lambda-wine-s01" {
  function_name                  = "wine-s01-scrape"
  architectures                  = ["x86_64"]
  image_uri                      = "${var.aws_account_number}.dkr.ecr.us-east-1.amazonaws.com/wine-s01-scrape:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::${var.aws_account_number}:role/iam-role-wine-lambda"
  timeout                        = "6"
  image_config { command = ["s01_wine_scrape/__main__.lambda_handler"] }
  tracing_config { mode = "PassThrough" }
  environment {
    variables = {
      INPUT_BUCKET  = "${var.aws_account_number}-wine-input"
      OUTPUT_BUCKET = "${var.aws_account_number}-wine-output"
    }
  }
}

resource "aws_lambda_function" "lambda-wine-s02" {
  function_name                  = "wine-s02-create-dataset"
  architectures                  = ["x86_64"]
  image_uri                      = "${var.aws_account_number}.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::${var.aws_account_number}:role/transcode-video"
  timeout                        = "6"
  image_config { command = ["transcode_video_python/__main__.lambda_handler"] }
  tracing_config { mode = "PassThrough" }
  environment {
    variables = {
      INPUT_BUCKET  = "${var.aws_account_number}-serverless-video-transcode-python"
      OUTPUT_BUCKET = "${var.aws_account_number}-serverless-video-transcode-python"
    }
  }
}

resource "aws_lambda_function" "lambda-wine-s03" {
  function_name                  = "wine-s03-fit"
  architectures                  = ["x86_64"]
  image_uri                      = "${var.aws_account_number}.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::${var.aws_account_number}:role/transcode-video"
  timeout                        = "6"
  image_config { command = ["transcode_video_python/__main__.lambda_handler"] }
  tracing_config { mode = "PassThrough" }
  environment {
    variables = {
      INPUT_BUCKET  = "${var.aws_account_number}-serverless-video-transcode-python"
      OUTPUT_BUCKET = "${var.aws_account_number}-serverless-video-transcode-python"
    }
  }
}

resource "aws_lambda_function" "lambda-wine-s04" {
  function_name                  = "wine-s04-predict"
  architectures                  = ["x86_64"]
  image_uri                      = "${var.aws_account_number}.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::${var.aws_account_number}:role/transcode-video"
  timeout                        = "6"
  image_config { command = ["transcode_video_python/__main__.lambda_handler"] }
  tracing_config { mode = "PassThrough" }
  environment {
    variables = {
      INPUT_BUCKET  = "${var.aws_account_number}-serverless-video-transcode-python"
      OUTPUT_BUCKET = "${var.aws_account_number}-serverless-video-transcode-python"
    }
  }
}

resource "aws_lambda_function" "lambda-wine-s05" {
  function_name                  = "wine-s05-score"
  architectures                  = ["x86_64"]
  image_uri                      = "${var.aws_account_number}.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest"
  memory_size                    = "1024"
  package_type                   = "Image"
  reserved_concurrent_executions = "-1"
  role                           = "arn:aws:iam::${var.aws_account_number}:role/transcode-video"
  timeout                        = "6"
  image_config { command = ["transcode_video_python/__main__.lambda_handler"] }
  tracing_config { mode = "PassThrough" }
  environment {
    variables = {
      INPUT_BUCKET  = "${var.aws_account_number}-serverless-video-transcode-python"
      OUTPUT_BUCKET = "${var.aws_account_number}-serverless-video-transcode-python"
    }
  }
}





