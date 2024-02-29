from datetime import datetime, timedelta

from s01_wine_scrape.issue_correction.save_missing_wines import main_correct_missing_parallel
from s01_wine_scrape.issue_detection.detect_missing import main_detect_missing_parallel

if __name__ == "__main__":
    today_date = datetime.now()
    yesterday_date = today_date - timedelta(days=1)
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')

    main_detect_missing_parallel(today_date_str)
    main_correct_missing_parallel(today_date_str)
    main_detect_missing_parallel(today_date_str)
