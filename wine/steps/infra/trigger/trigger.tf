variable "aws_account_number" {
  description = "AWS Account Number"
  type        = string
  default     = "064592191516"
}
resource "aws_s3_bucket_notification" "s3_trigger" {
  bucket = "${var.aws_account_number}-serverless-video-upload-python"
  lambda_function {
    lambda_function_arn = "arn:aws:lambda:us-east-1:${var.aws_account_number}:function:twentyfour-hour-video-python-transcode-video"
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
    filter_suffix       = ".mp4"
  }
}
resource "aws_lambda_permission" "aws_lambda_s3_trigger_permission" {
  statement_id  = "AllowS3UploadToInvoketwentyfourHourVideoPython"
  action        = "lambda:InvokeFunction"
  function_name = "function:twentyfour-hour-video-python-transcode-video"
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.aws_account_number}-serverless-video-upload-python"
}