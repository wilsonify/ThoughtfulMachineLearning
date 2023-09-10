resource "aws_s3_bucket" "s3-serverless-video-transcode-python" {
  bucket              = "064592191516-serverless-video-transcode-python"
  force_destroy       = false
  hosted_zone_id      = "Z3AQBSTGFYJSTF"
  object_lock_enabled = false
}

resource "aws_s3_bucket_request_payment_configuration" "s3-serverless-video-transcode-python-payer" {
  bucket = aws_s3_bucket.s3-serverless-video-transcode-python.id
  payer  = "BucketOwner"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "s3-serverless-video-transcode-python-sse" {
  bucket = aws_s3_bucket.s3-serverless-video-transcode-python.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = "arn:aws:kms:us-east-1:064592191516:alias/aws/s3"
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_versioning" "s3-serverless-video-transcode-python-versioning" {
  bucket = aws_s3_bucket.s3-serverless-video-transcode-python.id
  versioning_configuration {
    status = "Disabled"
  }
}

