# IAM Roles and Policies for wine rating predictor lambda

variable "aws_account_number" {
  description = "AWS Account Number"
  type        = string
  default     = "064592191516"
}

# IAM Policy Documents
data "aws_iam_policy_document" "lambda-policy-doc" {
  statement {
    actions   = ["s3:PutBucketNotification", "s3:GetBucketNotification"]
    effect    = "Allow"
    resources = [
      "arn:aws:s3:::${var.aws_account_number}-wine-input",
      "arn:aws:s3:::${var.aws_account_number}-wine-output"
    ]
  }
  statement {
    actions   = ["lambda:AddPermission", "lambda:RemovePermission"]
    effect    = "Allow"
    resources = ["arn:aws:lambda:us-east-1:${var.aws_account_number}:function/*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    effect    = "Allow"
    resources = ["arn:aws:logs:us-east-1:${var.aws_account_number}:log-group:/aws/lambda/*"]
  }

}


resource "aws_iam_policy" "lambda-policy" {
  name        = "wine-lambda-policy"
  description = "Policy for Custom Resources Lambda"
  policy      = data.aws_iam_policy_document.lambda-policy-doc.json
}


data "aws_iam_policy_document" "lambda-role-assume-policy" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# IAM Role for Custom Resources Lambda
resource "aws_iam_role" "wine-lambda" {
  name                 = "iam-role-wine-lambda"
  path                 = "/"
  assume_role_policy   = data.aws_iam_policy_document.lambda-role-assume-policy.json
  description          = "Allows Lambda functions to run wine service"
  max_session_duration = "3600"

}


# IAM Role Policy Attachment for Custom Resources Lambda
resource "aws_iam_role_policy_attachment" "wine-lambda-policy-attachment" {
  policy_arn = aws_iam_policy.lambda-policy.arn
  role       = aws_iam_role.wine-lambda.name
}


resource "aws_iam_role_policy_attachment" "wine-lambda-AWSLambdaExecute" {
  policy_arn = "arn:aws:iam::aws:policy/AWSLambdaExecute"
  role       = aws_iam_role.wine-lambda.name
}

