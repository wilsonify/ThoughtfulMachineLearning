# IAM Roles and Policies for 24hr video transcode lambda

variable "aws_account_number" {
  description = "AWS Account Number"
  type        = string
  default     = "064592191516"
}

# IAM Policy Document to allow the assumption of Roles
data "aws_iam_policy_document" "media-convert-role-assume-policy" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Service"
      identifiers = ["mediaconvert.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "custom-resources-lambda-role-assume-policy" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# IAM Policy Documents
data "aws_iam_policy_document" "custom-resources-lambda-policy-doc" {
  statement {
    actions   = ["s3:PutBucketNotification", "s3:GetBucketNotification"]
    effect    = "Allow"
    resources = ["arn:aws:s3:::${var.aws_account_number}-serverless-video-upload-python"]
  }

  statement {
    actions   = ["lambda:AddPermission", "lambda:RemovePermission"]
    effect    = "Allow"
    resources = ["arn:aws:lambda:us-east-1:${var.aws_account_number}:function/*"]
  }
}

data "aws_iam_policy_document" "lambda_cloudwatch_logs_policy_doc" {
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    effect    = "Allow"
    resources = ["arn:aws:logs:us-east-1:${var.aws_account_number}:log-group:/aws/lambda/your-lambda-function-name:*"]
  }
}

resource "aws_iam_policy" "custom-resources-lambda-policy" {
  name        = "custom-resources-lambda-policy"
  description = "Policy for Custom Resources Lambda"
  policy      = data.aws_iam_policy_document.custom-resources-lambda-policy-doc.json
}

resource "aws_iam_policy" "lambda_cloudwatch_logs_policy" {
  name        = "LambdaCloudWatchLogsPolicy"
  description = "IAM policy to allow Lambda to write logs to CloudWatch Logs"
  policy      = data.aws_iam_policy_document.lambda_cloudwatch_logs_policy_doc.json
}

# IAM Role for Media Convert
resource "aws_iam_role" "media-convert-role" {
  name                = "media-convert-role"
  path                = "/"
  assume_role_policy  = data.aws_iam_policy_document.media-convert-role-assume-policy.json
  description         = "Allows MediaConvert service to call S3 APIs and API Gateway on your behalf."
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  ]
  max_session_duration = "3600"
  tags                 = {
    app = "video-encoding-pipeline"
  }

}


# IAM Role for Custom Resources Lambda
resource "aws_iam_role" "custom-resources-lambda" {
  name                 = "twentyfour-hour-video-python-IamRoleCustomResources"
  path                 = "/"
  assume_role_policy   = data.aws_iam_policy_document.custom-resources-lambda-role-assume-policy.json
  description          = "Allows Lambda functions to call AWS video transcode services on your behalf."
  max_session_duration = "3600"

  tags = {
    STAGE = "dev"
  }


}

resource "aws_iam_role" "transcode-video" {
  assume_role_policy  = data.aws_iam_policy_document.custom-resources-lambda-role-assume-policy.json
  description         = "Allows Lambda functions to call AWS video transcode services on your behalf."
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AWSElementalMediaConvertFullAccess",
    "arn:aws:iam::aws:policy/AWSLambdaExecute"
  ]
  max_session_duration = "3600"
  name                 = "transcode-video"
  path                 = "/"

  tags = {
    app = "video-encoding-pipeline"
  }


}

# IAM Role Policy Attachment for Media Convert Role
resource "aws_iam_role_policy_attachment" "media-convert-role_AmazonS3FullAccess" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.media-convert-role.name
}

# IAM Role Policy Attachment for Custom Resources Lambda
resource "aws_iam_role_policy_attachment" "custom-resources-lambda-policy-attachment" {
  policy_arn = aws_iam_policy.custom-resources-lambda-policy.arn
  role       = aws_iam_role.custom-resources-lambda.name
}

# IAM Role Policy Attachment for Transcode Video Lambda
resource "aws_iam_role_policy_attachment" "transcode-video_AWSElementalMediaConvertFullAccess" {
  policy_arn = "arn:aws:iam::aws:policy/AWSElementalMediaConvertFullAccess"
  role       = aws_iam_role.transcode-video.name
}

resource "aws_iam_role_policy_attachment" "transcode-video_AWSLambdaExecute" {
  policy_arn = "arn:aws:iam::aws:policy/AWSLambdaExecute"
  role       = aws_iam_role.transcode-video.name
}

resource "aws_iam_role_policy_attachment" "lambda_cloudwatch_logs_attachment" {
  policy_arn = aws_iam_policy.lambda_cloudwatch_logs_policy.arn
  role       = aws_iam_role.transcode-video.name
}
