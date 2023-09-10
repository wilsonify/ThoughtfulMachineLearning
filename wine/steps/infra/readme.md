In this Terraform configuration:

* We define the AWS provider and region.
* We create an IAM role for the Lambda function and attach the necessary policy (AWSLambda_FullAccess).
* We define the Lambda function, specifying the code location, runtime, handler, and environment variables.
* We create an S3 bucket for video uploads.
* We grant the Lambda function permission to be triggered by S3 events.
* We configure an S3 bucket notification to trigger the Lambda function when objects are created in the specified prefix
  and with the ".jpg" suffix.

