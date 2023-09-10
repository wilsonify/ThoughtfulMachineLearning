# Indicate where to source the terraform module from.
terraform {
  source = "git::https://github.com/wilsonify/serverless-architectures-aws.git//p01-first-steps/c02-architectures/twentyfour-hour-video-python/infra/iam?ref=master"
}

# Indicate the input values to use for the variables of the module.
inputs = {
  tags = {
    DeployedBy  = "terragrunt"
    Environment = "dev"
  }
}