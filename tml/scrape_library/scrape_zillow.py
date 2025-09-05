import json
import pandas as pd
import random
import requests
import time
from bs4 import BeautifulSoup
from typing import Dict, List

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.zillow.com/homes/for_sale/{page}_p/"
OUTFILE = "properties-susquehannavalley.csv"

ATTRS_OF_INTEREST = [
    "list-card-addr",
    "list-card-type",
    "list-card-price",
    "list-card-details",
    "list-card-link",
]


def fetch_html(url: str, retries: int = 3, sleep_range=(1, 3)) -> BeautifulSoup:
    """Fetch a URL with retries and polite sleep."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(random.uniform(*sleep_range))
            return BeautifulSoup(resp.text, features="lxml")
        except Exception as e:
            logging.exception("Error fetching %s: %s", url, e)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return BeautifulSoup("", features="lxml")  # Explicit fallback return


def parse_property_card(card: BeautifulSoup) -> Dict[str, str]:
    """Extract property data from a search result card."""
    record = {}
    for y in card.find_all(attrs={"class": ["list-card-info"]}):
        for z in y.find_all(attrs={"class": ATTRS_OF_INTEREST}):
            record[z.attrs["class"][0]] = z.text
            if z.attrs["class"][0] == "list-card-link":
                record["list-card-link"] = z.attrs.get("href")
    return record


def enrich_with_details(record: Dict[str, str]) -> Dict[str, str]:
    """Fetch details from the property page and append to record."""
    link = record.get("list-card-link")
    if not link:
        return record

    try:
        detail_soup = fetch_html(link, sleep_range=(2, 5))
        facts = detail_soup.find_all(attrs={"class": "ds-home-fact-list"})
        for fact_list in facts:
            for item in fact_list:
                text = item.get_text(strip=True)
                if ":" in text:
                    label, value = text.split(":", 1)
                    record[label.strip()] = value.strip()
                else:
                    record.setdefault("meta", []).append(text)
    except Exception as e:
        record["detail_error"] = str(e)

    return record


def build_search_url(page: int) -> str:
    """Construct the Zillow search URL for a given page."""
    search_query_state = {
        "pagination": {"currentPage": page},
        "mapBounds": {
            "west": -77.16353057625066,
            "east": -76.40616058113348,
            "south": 40.79172057471558,
            "north": 41.214563845944205,
        },
        "isMapVisible": True,
        "mapZoom": 11,
        "filterState": {"sortSelection": {"value": "globalrelevanceex"}},
        "isListVisible": True,
    }
    return BASE_URL.format(page=page) + f"?searchQueryState={json.dumps(search_query_state)}"


def scrape_properties(pages: int = 20) -> List[Dict[str, str]]:
    """Scrape properties across multiple Zillow pages."""
    all_records = []
    for page in range(1, pages + 1):
        url = build_search_url(page)
        print(f"Scraping page {page}: {url}")
        soup = fetch_html(url)

        for card in soup.find_all(attrs={"class": ["list-card"]}):
            record = parse_property_card(card)
            record = enrich_with_details(record)
            all_records.append(record)

    return all_records


def main():
    records = scrape_properties(pages=20)
    df = pd.DataFrame(records)
    print(f"Scraped {len(df)} properties.")
    df.to_csv(OUTFILE, index=False)
    print(f"Saved to {OUTFILE}")


if __name__ == "__main__":
    main()
