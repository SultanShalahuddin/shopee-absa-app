"""
Scraping review Shopee Indonesia dari Google Play Store
Library: google-play-scraper (pip install google-play-scraper)

Catatan:
- App ID Shopee ID: com.shopee.id
- Kalau mau app lain, cek App ID-nya di URL Play Store:
  https://play.google.com/store/apps/details?id=<APP_ID>
"""

import pandas as pd
import time
from google_play_scraper import reviews, Sort

APP_ID = "com.shopee.id"
LANG = "id"
COUNTRY = "id"
TARGET_COUNT = 10000        # total review yang ingin diambil
BATCH_SIZE = 200           # jumlah review per request (max 200 per call)
OUTPUT_PATH = "data/raw/shopee_reviews_raw.csv"


def scrape_reviews(app_id, target_count, sort=Sort.NEWEST):
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < target_count:
        result, continuation_token = reviews(
            app_id,
            lang=LANG,
            country=COUNTRY,
            sort=sort,
            count=BATCH_SIZE,
            continuation_token=continuation_token,
        )

        if not result:
            print("Tidak ada review baru, berhenti lebih awal.")
            break

        all_reviews.extend(result)
        print(f"Total terkumpul: {len(all_reviews)}")

        # jeda kecil supaya tidak membebani endpoint terlalu cepat
        time.sleep(1)

        # kalau continuation_token None, artinya sudah habis semua review
        if continuation_token is None:
            break

    return all_reviews[:target_count]


def to_dataframe(raw_reviews):
    df = pd.DataFrame(raw_reviews)
    # kolom penting: reviewId, userName, content, score, thumbsUpCount,
    #                reviewCreatedVersion, at, appVersion
    keep_cols = [
        "reviewId",
        "content",
        "score",
        "thumbsUpCount",
        "at",
        "appVersion",
        "reviewCreatedVersion",
    ]
    return df[[c for c in keep_cols if c in df.columns]]


if __name__ == "__main__":
    print(f"Mulai scraping review untuk app: {APP_ID}")
    raw = scrape_reviews(APP_ID, TARGET_COUNT, sort=Sort.NEWEST)

    df = to_dataframe(raw)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSelesai. {len(df)} review disimpan ke {OUTPUT_PATH}")
    print(df.head())
