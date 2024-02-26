from pprint import pprint

from s01_wine_scrape.issue_correction.save_missing_wines import main_correct_missing_one_page
from s01_wine_scrape.issue_detection.detect_missing import main_detect_missing
from s01_wine_scrape.pa_wines.s01_save_pages_to_html import main_scrape_wine_pa
from s01_wine_scrape.pa_wines.s02_page_html_to_json import main_page_html_to_json
from s01_wine_scrape.pa_wines.s03_save_wines_to_html import main_scrape_wine_one_page
from s01_wine_scrape.pa_wines.s04_wine_html_to_json import main_scrape_one_wine
from s01_wine_scrape.pa_wines.s05_wine_json_to_csv import main_json_to_csv

available = {
    "main_scrape_wine_pa": main_scrape_wine_pa,
    "main_page_html_to_json": main_page_html_to_json,
    "main_scrape_one_wine": main_scrape_one_wine,
    "main_scrape_wine_one_page": main_scrape_wine_one_page,
    "main_json_to_csv": main_json_to_csv,
    "main_detect_missing": main_detect_missing,
    "main_correct_missing_one_page": main_correct_missing_one_page
}


def parse_event(event):
    parsed_event = {}
    for key, value in event.items():
        if hasattr(value, 'items') and callable(getattr(value, 'items')):
            parsed_event[key] = dict(value)
        else:
            parsed_event[key] = value
    return parsed_event


def lambda_handler(event, context):
    print("event")
    print(f"type(event) = {type(event)}")
    event_parsed = parse_event(event)
    pprint(event_parsed)
    print(f"type(event_parsed) = {type(event_parsed)}")

    print("context")
    pprint(vars(context))

    print("start main")
    strat_str = event.pop("strategy")
    strat_func = available[strat_str]

    print(f"start {strat_str}")
    strat_func(**event)
    print(f"done {strat_str}")

    print("done main")

    response = {
        "statusCode": 200,
        "body": "Success from s01_wine_scrape lambda"
    }
    return response
