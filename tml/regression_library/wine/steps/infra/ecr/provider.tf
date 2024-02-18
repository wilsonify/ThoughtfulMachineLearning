provider "aws" {
  profile = "064592191516"
}

terraform {
  required_version = ">1.5.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.76.1"
    }
  }
}
