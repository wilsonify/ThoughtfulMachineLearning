VERSION=0.2.6


docker-login-ecr:
	aws --profile 064592191516 ecr get-login-password | docker login --username AWS --password-stdin 064592191516.dkr.ecr.us-east-1.amazonaws.com

docker-build-base:
	docker build --progress=plain -t 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01-wine-scrape-base:$(VERSION) . -f Dockerfile-base

docker-push-base: docker-login-ecr
	docker push 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01_wine_scrape-base:$(VERSION)

docker-image:
	docker build --progress=plain -t 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01_wine_scrape:$(VERSION) .

docker-push: docker-login-ecr
	docker push 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01_wine_scrape:$(VERSION)

integrate: docker-image docker-push

deploy.json:
	aws lambda update-function-code --function-name s01-wine-scrape-lambda --image-uri 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01_wine_scrape:$(VERSION) > deploy.json

cicd: integrate deploy.json

docker-stop:
	docker stop tvp || true

docker-run: docker-stop docker-image
	docker run --rm --name tvp -p 32768:8080 -e AWS_DEFAULT_PROFILE=064592191516 -v ~/.aws:/root/.aws -d 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01_wine_scrape:$(VERSION)

docker-test-negative:
	curl "http://localhost:32768/2015-03-31/functions/function/invocations" -d '{}'

docker-test-positive:
	curl "http://localhost:32768/2015-03-31/functions/function/invocations" --data-binary '@test/example.json'

docker-test-positive-body:
	curl "http://localhost:32768/2015-03-31/functions/function/invocations" --data-binary '@test/example2.json'

docker-test-positive-sqs:
	curl "http://localhost:32768/2015-03-31/functions/function/invocations" --data-binary '@test/sqs.json'

check-infra:
	cd infra && terraform init && terraform validate && tflint

plan: check-infra
	cd infra && terraform plan

apply:
	cd infra && terragrunt --terragrunt-source-update plan && terragrunt apply -auto-approve