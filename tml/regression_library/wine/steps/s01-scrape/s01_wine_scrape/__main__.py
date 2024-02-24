from pprint import pprint

from s01_wine_scrape.scrape_wine import main_scrape_wine_pa


def lambda_handler(event, context):
    print("event")
    pprint(event)

    print("context")
    pprint(context)

    print("start main")
    main_scrape_wine_pa()
    print("done main")

    response = {
        "statusCode": 200,
        "body": "Success from mlflow-tf-s01-create-training-dataset lambda"
    }
    return response
