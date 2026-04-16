import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(r"e:\nyu homework\ra")
OUTPUT_DIR = BASE_DIR / "outputs"
HUMAN_VERIFY_PATH = BASE_DIR / "human-verify.txt"
INPUT_EVAL = OUTPUT_DIR / "evaluated_apps.csv"
FINAL_EVAL = OUTPUT_DIR / "evaluated_apps_final.csv"
FINAL_UNRESOLVED = OUTPUT_DIR / "unresolved_urls_final.csv"
SUBMISSION_EVAL = OUTPUT_DIR / "evaluated_apps_submission.csv"


EXPECTED_FIELDS = [
    "title",
    "app_type",
    "web_accessible",
    "web_url",
    "login_required",
    "login_methods",
    "age_verification_required",
    "age_verification_method",
    "subscription_required_for_long_chat",
    "all_features_available_without_subscription",
    "subscription_features",
    "subscription_cost",
    "languages_supported",
]


BOOL_TEXT = {"true", "false"}


def to_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def to_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def is_bool_token(token: str) -> bool:
    return token.strip().lower() in BOOL_TEXT


def is_url_token(token: str) -> bool:
    text = token.strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def clean_manual_value(field: str, value: str) -> Any:
    value = to_text(value)
    if not value:
        return ""
    if field in {
        "web_accessible",
        "login_required",
        "age_verification_required",
        "subscription_required_for_long_chat",
        "all_features_available_without_subscription",
    }:
        return to_bool(value)
    if field == "age_verification_method" and value.lower() == "flase":
        return ""
    return value


def parse_manual_line(line: str) -> Dict[str, Any]:
    parts = [part.strip() for part in line.rstrip().split("|")]
    while parts and parts[-1] == "":
        parts.pop()
    title = parts[0]
    tokens = parts[1:]

    result: Dict[str, Any] = {"title": title}
    if not tokens:
        return result

    result["app_type"] = to_text(tokens.pop(0))

    if tokens and is_bool_token(tokens[0]):
        result["web_accessible"] = to_bool(tokens.pop(0))
    elif tokens and is_url_token(tokens[0]):
        # Human verification omitted explicit web_accessible; infer True from supplied interactive URL.
        result["web_accessible"] = True

    if tokens and is_url_token(tokens[0]):
        result["web_url"] = to_text(tokens.pop(0))

    if tokens and is_bool_token(tokens[0]):
        result["login_required"] = to_bool(tokens.pop(0))

    if tokens and not is_bool_token(tokens[0]) and not is_url_token(tokens[0]):
        result["login_methods"] = to_text(tokens.pop(0))

    if tokens and is_bool_token(tokens[0]):
        result["age_verification_required"] = to_bool(tokens.pop(0))

    if tokens and not is_bool_token(tokens[0]) and not is_url_token(tokens[0]):
        result["age_verification_method"] = to_text(tokens.pop(0))

    if tokens and is_bool_token(tokens[0]):
        result["subscription_required_for_long_chat"] = to_bool(tokens.pop(0))

    if tokens and is_bool_token(tokens[0]):
        result["all_features_available_without_subscription"] = to_bool(tokens.pop(0))

    if tokens:
        result["subscription_features"] = to_text(tokens.pop(0))
    if tokens:
        result["subscription_cost"] = to_text(tokens.pop(0))
    if tokens:
        result["languages_supported"] = to_text(tokens.pop(0))

    for field, value in list(result.items()):
        if field == "title":
            continue
        result[field] = clean_manual_value(field, value)

    return result


def load_human_verify_records() -> List[Dict[str, Any]]:
    text = HUMAN_VERIFY_PATH.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    records = []
    for line in lines[1:]:
        records.append(parse_manual_line(line))
    return records


def apply_manual_record(df: pd.DataFrame, record: Dict[str, Any]) -> pd.DataFrame:
    mask = df["title"].astype(str) == record["title"]
    if not mask.any():
        return df

    for field, value in record.items():
        if field == "title" or field not in df.columns:
            continue
        df.loc[mask, field] = value

    df.loc[mask, "agent_unresolved"] = False
    df.loc[mask, "agent_unresolved_reason"] = ""
    df.loc[mask, "agent_notes"] = (
        df.loc[mask, "agent_notes"].fillna("").astype(str).map(lambda x: (x + " Manual verification applied.").strip())
    )
    return df


def build_unresolved_df(df: pd.DataFrame) -> pd.DataFrame:
    unresolved_df = df[df["agent_unresolved"] == True].copy()
    cols = ["title", "source_platform", "web_url", "agent_unresolved_reason", "agent_notes"]
    unresolved_df = unresolved_df[cols].rename(
        columns={
            "agent_unresolved_reason": "reason",
            "agent_notes": "notes",
        }
    )
    return unresolved_df


def fill_unknowns_for_submission(df: pd.DataFrame) -> pd.DataFrame:
    submission_df = df.copy()

    bool_like_cols = [
        "web_accessible",
        "login_required",
        "age_verification_required",
        "subscription_required_for_long_chat",
        "all_features_available_without_subscription",
    ]
    text_like_cols = [
        "web_url",
        "login_methods",
        "age_verification_method",
        "subscription_features",
        "subscription_cost",
        "languages_supported",
    ]

    for col in bool_like_cols:
        if col not in submission_df.columns:
            continue
        submission_df[col] = submission_df[col].map(
            lambda v: "unknown" if pd.isna(v) else ("True" if bool(v) else "False")
        )

    for col in text_like_cols:
        if col not in submission_df.columns:
            continue
        submission_df[col] = submission_df[col].map(
            lambda v: "unknown" if to_text(v) == "" else to_text(v)
        )

    return submission_df


def main() -> None:
    if not HUMAN_VERIFY_PATH.exists():
        raise FileNotFoundError(f"Missing human verify file: {HUMAN_VERIFY_PATH}")
    if not INPUT_EVAL.exists():
        raise FileNotFoundError(f"Missing evaluated file: {INPUT_EVAL}")

    df = pd.read_csv(INPUT_EVAL)
    manual_records = load_human_verify_records()

    for record in manual_records:
        df = apply_manual_record(df, record)

    unresolved_df = build_unresolved_df(df)
    submission_df = fill_unknowns_for_submission(df)

    final_eval_updated = True
    final_unresolved_updated = True
    try:
        df.to_csv(FINAL_EVAL, index=False)
    except PermissionError:
        final_eval_updated = False
    try:
        unresolved_df.to_csv(FINAL_UNRESOLVED, index=False)
    except PermissionError:
        final_unresolved_updated = False
    submission_df.to_csv(SUBMISSION_EVAL, index=False)

    print(f"Merged manual verification records: {len(manual_records)}")
    if final_eval_updated:
        print(f"Saved final evaluated results to: {FINAL_EVAL}")
    else:
        print(f"Skipped updating locked file: {FINAL_EVAL}")
    if final_unresolved_updated:
        print(f"Saved final unresolved results to: {FINAL_UNRESOLVED}")
    else:
        print(f"Skipped updating locked file: {FINAL_UNRESOLVED}")
    print(f"Saved submission-friendly evaluated results to: {SUBMISSION_EVAL}")
    print(f"Remaining unresolved rows: {len(unresolved_df)}")


if __name__ == "__main__":
    main()
