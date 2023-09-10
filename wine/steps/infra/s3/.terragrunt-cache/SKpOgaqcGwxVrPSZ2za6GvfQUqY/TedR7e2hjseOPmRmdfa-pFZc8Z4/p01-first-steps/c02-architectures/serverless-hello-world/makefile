init:
	sudo npm install -g serverless
	sudo npm install -g serverless-offline
	sls config credentials --provider aws --key <> --secret <>

hello-world/serverless.yml:
	sls create --template aws-nodejs --path hello-world

hello-world/.serverless/hello-world.zip:
	cd hello-world && sls deploy

invoke:
	cd hello-world && sls invoke --function hello

aws-nodejs-hello-world/serverless.yml:
	serverless create --template aws-nodejs --path aws-nodejs-hello-world

nodejs-docker-hello-world/serverless.yml:
	serverless create --template aws-nodejs-docker --path aws-nodejs-docker-hello-world

aws-nodejs-typescript-hello-world/serverless.yml:
	serverless create --template aws-nodejs-typescript --path aws-nodejs-typescript-hello-world

aws-python-hello-world/serverless.yml:
	serverless create --template aws-python --path aws-python-hello-world

aws-python3-hello-world/serverless.yml:
	serverless create --template aws-python3 --path aws-python3-hello-world

python-docker-hello-world/serverless.yml:
	serverless create --template aws-python-docker --path aws-python-docker-hello-world

aws-go-hello-world/serverless.yml:
	serverless create --template aws-go --path aws-go-hello-world

go-mod-hello-world/serverless.yml:
	serverless create --template aws-go-mod --path aws-go-mod-hello-world

spotinst-nodejs-hello-world/serverless.yml:
	serverless create --template spotinst-nodejs --path spotinst-nodejs-hello-world

spotinst-python-hello-world/serverless.yml:
	serverless create --template spotinst-python --path spotinst-python-hello-world
