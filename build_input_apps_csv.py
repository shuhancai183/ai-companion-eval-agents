import json
import csv
from pathlib import Path
from typing import Any, Dict, List


IOS_JSON = "app_store_apps_details.json"
ANDROID_JSON = "google_play_apps_details.json"
OUTPUT_CSV = "input_apps.csv"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_ios_rows(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for app in data.get("results", []):
        rows.append({
            "title": safe_str(app.get("title")),
            "source_platform": "ios",
            "app_id": safe_str(app.get("appId")),
            "store_url": safe_str(app.get("url")),
            "web_url": safe_str(app.get("developerWebsite")),
            "description": safe_str(app.get("description")),
            "content_rating": safe_str(app.get("contentRating")),
            "price_text": "Free" if app.get("free") else safe_str(app.get("price")),
            "offers_iap": "",  # App Store JSON here does not consistently expose it
            "developer": safe_str(app.get("developer")),
            "developer_website": safe_str(app.get("developerWebsite")),
            "languages_from_store": ",".join(app.get("languages", [])) if isinstance(app.get("languages"), list) else "",
        })
    return rows


def build_android_rows(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for app in data.get("results", []):
        rows.append({
            "title": safe_str(app.get("title")),
            "source_platform": "android",
            "app_id": safe_str(app.get("appId")),
            "store_url": safe_str(app.get("url")),
            "web_url": safe_str(app.get("developerWebsite")),
            "description": safe_str(app.get("description")),
            "content_rating": safe_str(app.get("contentRating")),
            "price_text": safe_str(app.get("priceText")),
            "offers_iap": "True" if app.get("offersIAP") else "False",
            "developer": safe_str(app.get("developer")),
            "developer_website": safe_str(app.get("developerWebsite")),
            "languages_from_store": "",  # Google Play JSON here usually doesn't expose a clean language list
        })
    return rows


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row.get("source_platform", ""),
            row.get("app_id", ""),
            row.get("title", "").strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def main():
    ios_data = load_json(IOS_JSON)
    android_data = load_json(ANDROID_JSON)

    rows = build_ios_rows(ios_data) + build_android_rows(android_data)
    rows = dedupe_rows(rows)

    fieldnames = [
        "title",
        "source_platform",
        "app_id",
        "store_url",
        "web_url",
        "description",
        "content_rating",
        "price_text",
        "offers_iap",
        "developer",
        "developer_website",
        "languages_from_store",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()