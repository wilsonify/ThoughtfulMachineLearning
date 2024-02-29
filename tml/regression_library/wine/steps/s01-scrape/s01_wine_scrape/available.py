from s01_wine_scrape.issue_correction.save_missing_wines import main_correct_missing_one_page
from s01_wine_scrape.issue_detection.detect_missing import main_detect_missing
from s01_wine_scrape.pa_wines.create.s01_save_pages_to_html import main_scrape_wine_pa
from s01_wine_scrape.pa_wines.create.s02_page_html_to_json import main_page_html_to_json
from s01_wine_scrape.pa_wines.create.s03_save_wines_to_html import main_scrape_wine_one_page
from s01_wine_scrape.pa_wines.create.s04_wine_html_to_json import main_scrape_one_wine
from s01_wine_scrape.pa_wines.create.s05_wine_json_to_csv import main_json_to_csv

available = {
    "main_scrape_wine_pa": main_scrape_wine_pa,
    "main_page_html_to_json": main_page_html_to_json,
    "main_scrape_one_wine": main_scrape_one_wine,
    "main_scrape_wine_one_page": main_scrape_wine_one_page,
    "main_json_to_csv": main_json_to_csv,
    "main_detect_missing": main_detect_missing,
    "main_correct_missing_one_page": main_correct_missing_one_page
}
