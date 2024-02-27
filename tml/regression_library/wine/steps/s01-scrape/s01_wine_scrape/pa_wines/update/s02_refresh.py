from s01_wine_scrape.issue_correction.save_missing_wines import main_correct_missing_parallel
from s01_wine_scrape.issue_detection.detect_missing import main_detect_missing_parallel

if __name__ == "__main__":
    main_detect_missing_parallel()
    main_correct_missing_parallel()
    main_detect_missing_parallel()
