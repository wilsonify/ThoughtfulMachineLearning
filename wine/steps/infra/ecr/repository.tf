variable "ecr_kms_key_arn" {
  type        = "string"
  default     = "arn:aws:kms:us-east-1:064592191516:key/83dec65c-2291-42e6-8116-187471e14532"
  description = "arn of key to use to encrypt ecr images"
}

resource "aws_ecr_repository" "ecr-s01-scrape" {
  name                 = "s01-scrape"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = "false" }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = var.ecr_kms_key_arn
  }
}

resource "aws_ecr_repository" "ecr-s02-create-dataset" {
  name                 = "s02-create-dataset"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = "false" }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = var.ecr_kms_key_arn
  }
}

resource "aws_ecr_repository" "ecr-s03-fit" {
  name                 = "s03-fit"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = "false" }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = var.ecr_kms_key_arn
  }
}

resource "aws_ecr_repository" "ecr-s04-predict" {
  name                 = "s04-predict"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = "false" }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = var.ecr_kms_key_arn
  }
}

resource "aws_ecr_repository" "ecr-s05-score" {
  name                 = "s05-score"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = "false" }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = var.ecr_kms_key_arn
  }
}