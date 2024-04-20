FROM 064592191516.dkr.ecr.us-east-1.amazonaws.com/s01-wine-scrape-base:0.2.5
COPY . /var/task
RUN pip install -r /var/task/requirements.txt
EXPOSE 8080
CMD [ "s01_wine_scrape/__main__.lambda_handler" ]
