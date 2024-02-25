from pprint import pprint

from s01_wine_scrape.pa_wines.s01_save_pages_to_html import main_scrape_wine_pa


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
        "body": "Success from s01_wine_scrape lambda"
    }
    return response
