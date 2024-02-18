FROM public.ecr.aws/lambda/python:3.8
COPY . /var/task
EXPOSE 8080
CMD [ "s01_wine_scrape/__main__.lambda_handler" ]
