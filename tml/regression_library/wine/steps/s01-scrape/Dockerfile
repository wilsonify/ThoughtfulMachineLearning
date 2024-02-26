FROM public.ecr.aws/lambda/python:3.10
COPY . /var/task
RUN pip install -r /var/task/requirements.txt
EXPOSE 8080
CMD [ "s01_wine_scrape/__main__.lambda_handler" ]
