init:
	sudo npm install -g serverless
	sudo npm install -g serverless-offline
	sls config credentials --provider aws --key <> --secret <>

serverless.yml:
	sls create --template aws-python3 --path .

.serverless/twentyfour-hour-video-python.zip:
	sls deploy

docker-image:
	docker build --progress=plain -t 064592191516.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest .

docker-login-ecr:
	aws --profile 064592191516 ecr get-login-password | docker login --username AWS --password-stdin 064592191516.dkr.ecr.us-east-1.amazonaws.com

docker-push:
	docker push 064592191516.dkr.ecr.us-east-1.amazonaws.com/transcode_video_python:latest

integrate: docker-image docker-login-ecr docker-push

check-infra:
	cd infra && terraform init && terraform validate && tflint

plan: check-infra
	cd infra && terraform plan

apply:
	terragrunt --terragrunt-source-update plan && terragrunt apply -auto-approve