import pandas as pd
from pathlib import Path
import shutil
from typing import Any, Optional


BASE_DIR = Path(r"e:\nyu homework\ra")
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_EVAL = OUTPUT_DIR / "evaluated_apps.csv"
OUTPUT_EVAL = OUTPUT_DIR / "evaluated_apps_repaired.csv"
OUTPUT_UNRESOLVED = OUTPUT_DIR / "unresolved_urls_repaired.csv"
FINAL_EVAL = OUTPUT_DIR / "evaluated_apps.csv"
FINAL_UNRESOLVED = OUTPUT_DIR / "unresolved_urls.csv"


def to_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
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
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def clean_bool_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].map(to_bool)


def clean_text_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].map(to_text)


def repair_row(row: pd.Series) -> pd.Series:
    web_accessible = to_bool(row.get("web_accessible"))
    login_required = to_bool(row.get("login_required"))
    age_required = to_bool(row.get("age_verification_required"))
    unresolved = to_bool(row.get("agent_unresolved"))
    unresolved_reason = to_text(row.get("agent_unresolved_reason"))

    # Normalize text-like fields.
    for col in [
        "web_url",
        "login_methods",
        "age_verification_method",
        "subscription_features",
        "subscription_cost",
        "languages_supported",
        "agent_unresolved_reason",
        "agent_notes",
    ]:
        if col in row.index:
            row[col] = to_text(row.get(col))

    # Rule: if not web accessible, web-only interaction fields should not be forced.
    if web_accessible is False:
        row["login_required"] = None
        row["login_methods"] = ""

    # Rule: if login is not required/unknown, no login methods.
    login_required = to_bool(row.get("login_required"))
    if login_required is not True:
        row["login_methods"] = ""

    # Rule: if age verification is not required/unknown, no method text.
    if age_required is not True:
        row["age_verification_method"] = ""

    # Rule: if web accessible is true, login_required is a core field.
    if web_accessible is True and login_required is None:
        row["agent_unresolved"] = True
        row["agent_unresolved_reason"] = unresolved_reason or "Missing core field: login_required for a web-accessible app."
    else:
        # Only keep unresolved for strong core blockers.
        strong_markers = [
            "cloudflare",
            "blocked",
            "missing core field",
            "missing core fields",
            "per-app timeout",
            "page timeout",
        ]
        if unresolved is True and any(marker in unresolved_reason.lower() for marker in strong_markers):
            row["agent_unresolved"] = True
        else:
            row["agent_unresolved"] = False
            row["agent_unresolved_reason"] = ""

    return row


def build_unresolved_df(df: pd.DataFrame) -> pd.DataFrame:
    unresolved_df = df[df["agent_unresolved"] == True].copy()
    cols = ["title", "source_platform", "web_url", "agent_unresolved_reason", "agent_notes"]
    available_cols = [c for c in cols if c in unresolved_df.columns]
    unresolved_df = unresolved_df[available_cols].rename(
        columns={
            "agent_unresolved_reason": "reason",
            "agent_notes": "notes",
        }
    )
    return unresolved_df


def main() -> None:
    if not INPUT_EVAL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_EVAL}")

    df = pd.read_csv(INPUT_EVAL)

    for col in [
        "web_accessible",
        "login_required",
        "age_verification_required",
        "subscription_required_for_long_chat",
        "all_features_available_without_subscription",
        "agent_unresolved",
    ]:
        clean_bool_column(df, col)

    for col in [
        "web_url",
        "login_methods",
        "age_verification_method",
        "subscription_features",
        "subscription_cost",
        "languages_supported",
        "agent_unresolved_reason",
        "agent_notes",
    ]:
        clean_text_column(df, col)

    df = df.apply(repair_row, axis=1)
    unresolved_df = build_unresolved_df(df)

    df.to_csv(OUTPUT_EVAL, index=False)
    unresolved_df.to_csv(OUTPUT_UNRESOLVED, index=False)
    shutil.copyfile(OUTPUT_EVAL, FINAL_EVAL)
    shutil.copyfile(OUTPUT_UNRESOLVED, FINAL_UNRESOLVED)

    print(f"Saved repaired evaluated results to: {OUTPUT_EVAL}")
    print(f"Saved repaired unresolved results to: {OUTPUT_UNRESOLVED}")
    print(f"Updated final evaluated results at: {FINAL_EVAL}")
    print(f"Updated final unresolved results at: {FINAL_UNRESOLVED}")
    print(f"Repaired rows: {len(df)}")
    print(f"Unresolved rows after repair: {len(unresolved_df)}")


if __name__ == "__main__":
    main()
