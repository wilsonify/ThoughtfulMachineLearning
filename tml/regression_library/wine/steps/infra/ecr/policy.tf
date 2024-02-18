resource "aws_ecr_repository_policy" "ecr_s01_policy" {
  policy     = data.aws_iam_policy_document.ecr-wine-policy-doc.json
  repository = aws_ecr_repository.ecr-s01-scrape.name
}

resource "aws_ecr_repository_policy" "ecr_s02_policy" {
  policy     = data.aws_iam_policy_document.ecr-wine-policy-doc.json
  repository = aws_ecr_repository.ecr-s02-create-dataset.name
}

resource "aws_ecr_repository_policy" "ecr_s03_policy" {
  policy     = data.aws_iam_policy_document.ecr-wine-policy-doc.json
  repository = aws_ecr_repository.ecr-s03-fit.name
}

resource "aws_ecr_repository_policy" "ecr_s04_policy" {
  policy     = data.aws_iam_policy_document.ecr-wine-policy-doc.json
  repository = aws_ecr_repository.ecr-s04-predict.name
}

resource "aws_ecr_repository_policy" "ecr_s05_policy" {
  policy     = data.aws_iam_policy_document.ecr-wine-policy-doc.json
  repository = aws_ecr_repository.ecr-s05-score.name
}