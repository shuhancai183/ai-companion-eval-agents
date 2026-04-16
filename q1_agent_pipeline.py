import os
import json
import time
import hashlib
import random
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urljoin, urlparse

import pandas as pd
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ============================================================
# Config
# ============================================================
INPUT_CSV = "input_apps.csv"
OUTPUT_DIR = "outputs"
OUT_EVAL = os.path.join(OUTPUT_DIR, "evaluated_apps.csv")
OUT_UNRESOLVED = os.path.join(OUTPUT_DIR, "unresolved_urls.csv")
CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint.jsonl")
ENV_FILE = ".env"

MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = "https://api.chatanywhere.org/v1"
HEADLESS = True
PAGE_TIMEOUT_MS = 20000
MAX_VISIBLE_TEXT_CHARS = 12000
MAX_STEPS = 4
OPEN_PAGE_SLEEP_SECONDS = 1.0
POST_CLICK_SLEEP_SECONDS = 1.0
POST_SEND_SLEEP_SECONDS = 2.0
SLEEP_BETWEEN_APPS = 0.2
RUN_FIRST_N = None
LLM_TIMEOUT_SECONDS = 60
PER_APP_TIMEOUT_SECONDS = 180
NAVIGATION_WAIT_UNTIL = "commit"
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

USE_RULE_TOOL = True  # optional


# ============================================================
# Rules
# ============================================================
RULES = {
    "app_type": """
Classify as one of:
- companion: primarily marketed around social, emotional, relational, romantic, roleplay, or character-based engagement.
- general_purpose: broad general-purpose LLM assistants such as ChatGPT, Claude, Gemini, Grok, DeepSeek, Perplexity, Copilot, etc.
- mixed: clearly offers both companion-style relational engagement and broad general-purpose assistant capabilities.
- other: neither of the above, including task-specific tools such as study/homework/productivity apps.

Important:
- Companion means the PRIMARY positioning is relational/social engagement.
- General-purpose assistants must NOT be classified as companion.
- Task-specific apps must NOT be classified as companion or general_purpose.
- If an app offers both companion and general-purpose capabilities, classify it as mixed.
""".strip(),
    "web_accessible": """
True only if the AI can actually be interacted with via a website.
False if the website is only a homepage, download page, or marketing page without usable AI interaction.
""".strip(),
    "web_url": """
If web_accessible is true, provide the interactive web URL.
If not interactable on web, leave empty or unresolved.
""".strip(),
    "login_required": """
True if login is required to interact with the AI characters.
Also treat as True if the website allows only limited interaction before requiring login.
Only mark False if the site supports very long conversations without login.
""".strip(),
    "login_methods": """
If login is required, list supported methods such as email/password, Google, Apple, Facebook, TikTok, etc.
""".strip(),
    "age_verification_required": """
True if the platform enforces age gating, including self-declaration, birthdate gates, or stronger verification.
""".strip(),
    "age_verification_method": """
If age verification is required, describe the mechanism such as self-declaration, DOB gate, ID upload.
""".strip(),
    "subscription_required_for_long_chat": """
True if a paid subscription is required to sustain longer conversations.
Our target is effectively unlimited chat, or at least a few thousand messages in one conversation.
""".strip(),
    "all_features_available_without_subscription": """
True only if all core features are accessible without paying.
""".strip(),
    "subscription_features": """
If a subscription exists, describe added features such as unlimited messaging, premium characters, faster responses, memory, voice, NSFW access.
""".strip(),
    "subscription_cost": """
Provide monthly subscription pricing including currency when available.
If only weekly/annual pricing is visible, record what is visible and mention limitation in notes.
""".strip(),
    "languages_supported": """
List platform languages available to users, not model-internal multilingual ability.
Use short standardized identifiers when possible (e.g. en, fr, de), or full names if necessary.
""".strip(),
}


def get_rule(field_name: str) -> str:
    return RULES.get(field_name, "No rule found.")


# ============================================================
# Browser tool layer
# ============================================================
def clean_text(text: str, max_len: int = MAX_VISIBLE_TEXT_CHARS) -> str:
    if not text:
        return ""
    return " ".join(text.split())[:max_len]


class BrowserTools:
    def __init__(self, page):
        self.page = page

    def open_page(self, url: str) -> Dict[str, Any]:
        self.page.goto(url, wait_until=NAVIGATION_WAIT_UNTIL, timeout=PAGE_TIMEOUT_MS)
        time.sleep(OPEN_PAGE_SLEEP_SECONDS)
        return {"ok": True, "message": f"Opened {url}"}

    def get_page_state(self) -> Dict[str, Any]:
        try:
            title = self.page.title()
        except Exception:
            title = ""

        current_url = self.page.url

        visible_text = ""
        try:
            visible_text = self.page.locator("body").inner_text(timeout=5000)
        except Exception:
            pass

        visible_text = clean_text(visible_text)

        clickables = []
        try:
            elems = self.page.locator("button, a, [role='button'], input[type='button'], input[type='submit']")
            n = min(elems.count(), 80)
            for i in range(n):
                try:
                    txt = elems.nth(i).inner_text(timeout=500).strip()
                    if txt:
                        clickables.append(txt)
                except Exception:
                    continue
        except Exception:
            pass

        inputs = []
        try:
            elems = self.page.locator("textarea, input[type='text'], input:not([type]), [contenteditable='true']")
            n = min(elems.count(), 20)
            for i in range(n):
                try:
                    inputs.append({
                        "placeholder": elems.nth(i).get_attribute("placeholder") or "",
                        "aria_label": elems.nth(i).get_attribute("aria-label") or "",
                        "type": elems.nth(i).get_attribute("type") or "",
                    })
                except Exception:
                    continue
        except Exception:
            pass

        return {
            "page_title": title,
            "current_url": current_url,
            "visible_text": visible_text,
            "clickable_texts": clickables[:50],
            "input_elements": inputs[:10],
        }

    def click_text(self, text: str) -> Dict[str, Any]:
        try:
            locator = self.page.locator(f"text={text}").first
            locator.click(timeout=3000)
            time.sleep(POST_CLICK_SLEEP_SECONDS)
            return {"ok": True, "message": f"Clicked '{text}'"}
        except Exception as e:
            return {"ok": False, "message": f"Click failed for '{text}': {e}"}

    def click_matching_text(self, candidates: List[str], exact: bool = False) -> Dict[str, Any]:
        clickables = self.page.locator("button, a, [role='button'], input[type='button'], input[type='submit'], label")
        try:
            count = min(clickables.count(), 100)
        except Exception as e:
            return {"ok": False, "message": f"Could not enumerate clickable elements: {e}"}

        lowered_candidates = [c.strip().lower() for c in candidates if c and c.strip()]
        for i in range(count):
            try:
                elem = clickables.nth(i)
                text = (elem.inner_text(timeout=500) or "").strip()
                if not text:
                    text = (elem.get_attribute("aria-label") or "").strip()
                hay = text.lower()
                if not hay:
                    continue
                matched = any(hay == candidate for candidate in lowered_candidates) if exact else any(candidate in hay for candidate in lowered_candidates)
                if not matched:
                    continue
                elem.click(timeout=1500)
                time.sleep(POST_CLICK_SLEEP_SECONDS)
                return {"ok": True, "message": f"Clicked matching element '{text}'"}
            except Exception:
                continue

        return {"ok": False, "message": "No matching clickable element found"}

    def choose_random_option(self, candidates: List[str]) -> Dict[str, Any]:
        clickables = self.page.locator("button, a, [role='button'], input[type='button'], input[type='submit'], label")
        matches: List[Any] = []
        labels: List[str] = []
        lowered_candidates = [c.strip().lower() for c in candidates if c and c.strip()]

        try:
            count = min(clickables.count(), 100)
        except Exception as e:
            return {"ok": False, "message": f"Could not enumerate option elements: {e}"}

        for i in range(count):
            try:
                elem = clickables.nth(i)
                text = (elem.inner_text(timeout=500) or "").strip()
                if not text:
                    text = (elem.get_attribute("aria-label") or "").strip()
                hay = text.lower()
                if not hay:
                    continue
                if any(candidate in hay for candidate in lowered_candidates):
                    matches.append(elem)
                    labels.append(text)
            except Exception:
                continue

        if not matches:
            return {"ok": False, "message": "No selectable option found"}

        choice_index = random.randrange(len(matches))
        try:
            matches[choice_index].click(timeout=1500)
            time.sleep(POST_CLICK_SLEEP_SECONDS)
            return {"ok": True, "message": f"Randomly selected option '{labels[choice_index]}'"}
        except Exception as e:
            return {"ok": False, "message": f"Random option click failed: {e}"}

    def fill_search_box(self, query: str) -> Dict[str, Any]:
        search_selectors = [
            "input[type='search']",
            "input[placeholder*='search' i]",
            "input[aria-label*='search' i]",
            "textarea[placeholder*='search' i]",
            "textarea[aria-label*='search' i]",
            "input[type='text']",
            "input:not([type])",
        ]

        for sel in search_selectors:
            try:
                elems = self.page.locator(sel)
                count = min(elems.count(), 10)
                for i in range(count):
                    elem = elems.nth(i)
                    placeholder = (elem.get_attribute("placeholder") or "").lower()
                    aria_label = (elem.get_attribute("aria-label") or "").lower()
                    input_type = (elem.get_attribute("type") or "").lower()
                    looks_like_search = (
                        "search" in placeholder
                        or "search" in aria_label
                        or input_type == "search"
                        or sel in {"input[type='text']", "input:not([type])"}
                    )
                    if not looks_like_search:
                        continue
                    elem.click(timeout=1000)
                    elem.fill(query, timeout=2000)
                    elem.press("Enter")
                    time.sleep(POST_CLICK_SLEEP_SECONDS)
                    return {"ok": True, "message": f"Searched for '{query}'"}
            except Exception:
                continue

        return {"ok": False, "message": "No search box found"}

    def search_first_result_url(self, query: str) -> Dict[str, Any]:
        search_engines = [
            f"https://html.duckduckgo.com/html/?q={quote(query)}",
            f"https://www.google.com/search?q={quote(query)}",
        ]

        for search_url in search_engines:
            try:
                self.page.goto(search_url, wait_until=NAVIGATION_WAIT_UNTIL, timeout=PAGE_TIMEOUT_MS)
                time.sleep(OPEN_PAGE_SLEEP_SECONDS)
            except Exception:
                continue

            anchors = self.page.locator("a")
            try:
                count = min(anchors.count(), 80)
            except Exception:
                continue

            for i in range(count):
                try:
                    href = anchors.nth(i).get_attribute("href") or ""
                    target_url = normalize_search_result_url(href, self.page.url)
                    if not target_url:
                        continue
                    self.page.goto(target_url, wait_until=NAVIGATION_WAIT_UNTIL, timeout=PAGE_TIMEOUT_MS)
                    time.sleep(OPEN_PAGE_SLEEP_SECONDS)
                    return {"ok": True, "message": f"Opened search result {target_url}", "url": target_url}
                except Exception:
                    continue

        return {"ok": False, "message": "No usable search result found", "url": ""}

    def send_message(self, msg: str) -> Dict[str, Any]:
        input_selectors = [
            "textarea",
            "input[type='text']",
            "input:not([type])",
            "[contenteditable='true']",
        ]

        input_box = None
        for sel in input_selectors:
            try:
                loc = self.page.locator(sel).first
                if loc.count() > 0:
                    input_box = loc
                    break
            except Exception:
                continue

        if input_box is None:
            return {"ok": False, "message": "No input box found"}

        try:
            input_box.click(timeout=2000)
            input_box.fill(msg, timeout=3000)
        except Exception as e:
            return {"ok": False, "message": f"Could not fill input: {e}"}

        send_attempted = False

        # Try obvious send buttons first
        button_candidates = [
            "button:has-text('Send')",
            "button[type='submit']",
            "[role='button'][aria-label*='send' i]",
            "button",
        ]

        for sel in button_candidates:
            try:
                elems = self.page.locator(sel)
                n = min(elems.count(), 10)
                for i in range(n):
                    try:
                        txt = elems.nth(i).inner_text(timeout=500).strip().lower()
                        if sel == "button" and txt not in {"send", "submit", "chat", "start", "continue"}:
                            continue
                        elems.nth(i).click(timeout=1000)
                        send_attempted = True
                        break
                    except Exception:
                        continue
                if send_attempted:
                    break
            except Exception:
                continue

        if not send_attempted:
            try:
                input_box.press("Enter")
                send_attempted = True
            except Exception:
                pass

        if not send_attempted:
            return {"ok": False, "message": "Could not trigger send"}

        time.sleep(POST_SEND_SLEEP_SECONDS)
        return {"ok": True, "message": "Message sent"}

    def screenshot(self, path: str) -> Dict[str, Any]:
        try:
            self.page.screenshot(path=path, full_page=True)
            return {"ok": True, "message": f"Saved screenshot to {path}"}
        except Exception as e:
            return {"ok": False, "message": f"Screenshot failed: {e}"}


# ============================================================
# LLM schemas
# ============================================================
class Action(BaseModel):
    action: str = Field(description="One of open_page, get_page_state, click_text, send_message, consult_rule, finish")
    argument: Optional[str] = ""
    done: bool = False
    reason: str = ""


class FinalResult(BaseModel):
    app_type: Optional[str] = None
    web_accessible: Optional[bool] = None
    web_url: str = ""
    login_required: Optional[bool] = None
    login_methods: List[str] = Field(default_factory=list)
    age_verification_required: Optional[bool] = None
    age_verification_method: str = ""
    subscription_required_for_long_chat: Optional[bool] = None
    all_features_available_without_subscription: Optional[bool] = None
    subscription_features: str = ""
    subscription_cost: str = ""
    languages_supported: List[str] = Field(default_factory=list)
    confidence: str = "medium"
    unresolved: bool = False
    unresolved_reason: str = ""
    notes: str = ""


class AppProcessingTimeout(Exception):
    pass


# ============================================================
# Prompts
# ============================================================
SYSTEM_RULES_TEXT = """
You are evaluating apps for a research dataset on AI companion platforms.

Field definitions:

1. app_type
- companion: primarily marketed around social, emotional, relational, romantic, roleplay, or character-based engagement.
- general_purpose: broad general-purpose LLM assistants such as ChatGPT, Claude, Gemini, Grok, DeepSeek, Perplexity, Copilot, etc.
- mixed: clearly offers both companion-style relational engagement and broad general-purpose assistant capabilities.
- other: neither of the above, including task-specific tools such as study/homework/productivity apps.

Important:
- Companion means the PRIMARY positioning is relational/social engagement.
- General-purpose assistants must NOT be classified as companion.
- Task-specific apps must NOT be classified as companion or general_purpose.
- If an app offers both companion and general-purpose capabilities, classify it as mixed.

2. web_accessible
True only if the AI can actually be interacted with via a website.
False if the website is only a homepage, download page, or marketing page without usable AI interaction.

3. web_url
If web_accessible is true, provide the interactive web URL.

4. login_required
True if login is required to interact with the AI characters.
Also treat as True if the website allows only limited interaction before requiring login.
Only mark False if the site supports very long conversations without login.

5. login_methods
If login is required, list supported methods such as email/password, Google, Apple, Facebook, TikTok, etc.

6. age_verification_required
True if the platform enforces age gating, including self-declaration, birthdate gates, or stronger verification.

7. age_verification_method
If age verification is required, describe the mechanism such as self-declaration, DOB gate, ID upload.

8. subscription_required_for_long_chat
True if a paid subscription is required to sustain longer conversations.
Our target is effectively unlimited chat, or at least a few thousand messages in one conversation.

9. all_features_available_without_subscription
True only if all core features are accessible without paying.

10. subscription_features
If a subscription exists, describe added features such as unlimited messaging, premium characters, faster responses, memory, voice, NSFW access.

11. subscription_cost
Provide monthly subscription pricing including currency when available.
If only weekly/annual pricing is visible, record what is visible and mention limitations in notes.

12. languages_supported
List platform languages available to users, not model-internal multilingual ability.
Use short standardized identifiers when possible (e.g. en, fr, de), or full names if necessary.

Conservatism rules:
- Be conservative.
- Do not guess unsupported facts.
- If evidence is insufficient, mark unresolved=true.
- If website interaction is ambiguous, do not assume web_accessible=true.
- If login appears after a few messages, treat login_required=true.
- Missing evidence for secondary fields such as subscription cost or supported languages does not by itself require unresolved=true if the main classification is otherwise supported.
- If age gating is not clearly visible, do not invent it.
- If the interaction history shows age-selection, DOB entry, or adult-confirmation steps, treat that as evidence for age_verification_required=true.
- If the interaction history shows profile setup choices such as gender selection, include that evidence in notes even though gender is not a required output field.
""".strip()

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_RULES_TEXT
        + "\n\n"
        + """You control a browser agent with tools:
- open_page(url)
- get_page_state()
- click_text(text)
- send_message(text)
- consult_rule(field_name)

Your goal is to gather enough evidence for the final structured output.
Use minimal actions. Prefer finish when evidence is sufficient.
Output JSON only.
"""
    ),
    (
        "human",
        """App metadata:
{app_metadata}

Current browser observation:
{state}

Action history:
{history}

Decide the next step.

Return JSON:
{{
  "action": "...",
  "argument": "...",
  "done": false,
  "reason": "..."
}}

Allowed actions:
- get_page_state
- click_text
- send_message
- consult_rule
- finish
"""
    )
])

FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_RULES_TEXT
        + "\n\n"
        + "Summarize the app into a final structured JSON. Output JSON only."
    ),
    (
        "human",
        """App metadata:
{app_metadata}

Browser/action history:
{history}

Return JSON with keys exactly:
app_type, web_accessible, web_url, login_required, login_methods,
age_verification_required, age_verification_method,
subscription_required_for_long_chat, all_features_available_without_subscription,
subscription_features, subscription_cost, languages_supported,
confidence, unresolved, unresolved_reason, notes
"""
    )
])


# ============================================================
# Helper functions
# ============================================================
def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_within_app_timeout(start_time: float, title: str) -> None:
    elapsed = time.time() - start_time
    if elapsed > PER_APP_TIMEOUT_SECONDS:
        raise AppProcessingTimeout(
            f"per-app timeout after {elapsed:.1f}s while processing '{title}'"
        )


def load_dotenv_file(path: str = ENV_FILE) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def normalize_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        x = v.strip().lower()
        if x in {"true", "yes", "y", "1"}:
            return True
        if x in {"false", "no", "n", "0"}:
            return False
    return None


def coerce_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, list):
        return ", ".join(coerce_str(item) for item in v if coerce_str(item))
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return default
    return str(v).strip()


def coerce_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [item for item in (coerce_str(x) for x in v) if item]
    if isinstance(v, str):
        text = v.strip()
        if not text:
            return []
        for sep in [";", ",", "/", "|"]:
            if sep in text:
                return [item for item in (part.strip() for part in text.split(sep)) if item]
        return [text]
    coerced = coerce_str(v)
    return [coerced] if coerced else []


def normalize_confidence(v: Any) -> str:
    if isinstance(v, (int, float)):
        if v >= 0.85:
            return "high"
        if v >= 0.5:
            return "medium"
        return "low"
    text = coerce_str(v).lower()
    if text in {"low", "medium", "high"}:
        return text
    if text in {"certain", "very high"}:
        return "high"
    if text in {"uncertain", "unsure"}:
        return "low"
    return "medium"


def normalize_final_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    normalized["app_type"] = coerce_str(normalized.get("app_type")) or "other"
    normalized["web_accessible"] = coerce_bool(normalized.get("web_accessible"))
    normalized["web_url"] = coerce_str(normalized.get("web_url"))
    normalized["login_required"] = coerce_bool(normalized.get("login_required"))
    normalized["login_methods"] = coerce_str_list(normalized.get("login_methods"))
    normalized["age_verification_required"] = coerce_bool(normalized.get("age_verification_required"))
    normalized["age_verification_method"] = coerce_str(normalized.get("age_verification_method"))
    normalized["subscription_required_for_long_chat"] = coerce_bool(normalized.get("subscription_required_for_long_chat"))
    normalized["all_features_available_without_subscription"] = coerce_bool(normalized.get("all_features_available_without_subscription"))
    normalized["subscription_features"] = coerce_str(normalized.get("subscription_features"))
    normalized["subscription_cost"] = coerce_str(normalized.get("subscription_cost"))
    normalized["languages_supported"] = coerce_str_list(normalized.get("languages_supported"))
    normalized["confidence"] = normalize_confidence(normalized.get("confidence"))
    unresolved = coerce_bool(normalized.get("unresolved"))
    normalized["unresolved"] = False if unresolved is None else unresolved
    normalized["unresolved_reason"] = coerce_str(normalized.get("unresolved_reason"))
    normalized["notes"] = coerce_str(normalized.get("notes"))
    return normalized


def parse_final_result(raw_text: str) -> FinalResult:
    raw_payload = json.loads(raw_text)
    normalized_payload = normalize_final_result_payload(raw_payload)
    return FinalResult(**normalized_payload)


def normalize_key_part(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_search_result_url(href: str, current_page_url: str = "") -> str:
    if not href:
        return ""
    if href.startswith("/url?"):
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        target = query.get("q", [""])[0]
        href = target or ""
    elif "duckduckgo.com/l/?" in href:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        target = query.get("uddg", [""])[0]
        href = target or ""
    elif href.startswith("/"):
        href = urljoin(current_page_url or "https://www.google.com", href)

    if not href:
        return ""

    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    blocked_hosts = ["google.", "duckduckgo.com", "bing.com", "search.yahoo.com"]
    if any(blocked in parsed.netloc.lower() for blocked in blocked_hosts):
        return ""
    return href


def normalize_input_url(url: Any) -> str:
    text = str(url or "").strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def is_invalid_or_missing_url(url: str) -> bool:
    normalized = normalize_input_url(url)
    if not normalized:
        return True
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return True
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host.startswith("api.") or path.startswith("/api"):
        return True
    return False


def state_looks_invalid_url(state: Dict[str, Any]) -> bool:
    title = str(state.get("page_title", "")).lower()
    current_url = str(state.get("current_url", "")).lower()
    visible_text = str(state.get("visible_text", "")).lower()
    corpus = f"{title} {current_url} {visible_text}"
    invalid_markers = [
        "404", "page not found", "not found", "this site can't be reached",
        "dns_probe", "err_name_not_resolved", "server not found",
    ]
    return any(marker in corpus for marker in invalid_markers)


def has_history_signal(history: List[Dict[str, Any]], keywords: List[str]) -> bool:
    text = json.dumps(history, ensure_ascii=False).lower()
    return any(keyword in text for keyword in keywords)


def has_metadata_signal(row: Dict[str, Any], keywords: List[str]) -> bool:
    fields = [
        str(row.get("description", "")),
        str(row.get("content_rating", "")),
        str(row.get("price_text", "")),
        str(row.get("offers_iap", "")),
        str(row.get("languages_from_store", "")),
    ]
    text = " ".join(fields).lower()
    return any(keyword in text for keyword in keywords)


def has_positive_login_evidence(row: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    keywords = ["login", "log in", "sign in", "continue with", "google", "apple", "facebook", "verification code", "email/password"]
    return has_history_signal(history, keywords) or has_metadata_signal(row, keywords)


def has_negative_login_evidence(history: List[Dict[str, Any]]) -> bool:
    return has_history_signal(history, ['"tool": "send_message"', '"ok": true'])


def has_age_evidence(row: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    keywords = ["18+", "adult", "your age", "age gate", "date of birth", "dob", "birthday", "self-declaration"]
    return has_history_signal(history, keywords) or has_metadata_signal(row, keywords)


def has_subscription_evidence(row: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    keywords = [
        "subscription", "subscribe", "premium", "pro", "plus", "unlimited chat",
        "unlimited messaging", "free trial", "monthly", "weekly", "annual", "$",
    ]
    offers_iap = str(row.get("offers_iap", "")).strip().lower()
    return offers_iap in {"true", "1", "yes"} or has_history_signal(history, keywords) or has_metadata_signal(row, keywords)


def has_language_evidence(row: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    if normalize_input_url(row.get("languages_from_store", "")):
        return True
    keywords = ["language", "english", "japanese", "korean", "chinese", "french", "german", "spanish"]
    return has_history_signal(history, keywords) or has_metadata_signal(row, keywords)


def apply_core_unresolved_policy(row: Dict[str, Any], result: FinalResult) -> FinalResult:
    core_missing: List[str] = []

    if result.app_type not in {"companion", "general_purpose", "mixed", "other"}:
        core_missing.append("app_type")
    if result.web_accessible is None:
        core_missing.append("web_accessible")
    if result.web_accessible is True and not coerce_str(result.web_url):
        core_missing.append("web_url")
    if result.web_accessible is True and result.login_required is None:
        core_missing.append("login_required")

    if core_missing:
        result.unresolved = True
        if not result.unresolved_reason:
            result.unresolved_reason = f"Missing core fields: {', '.join(core_missing)}"
        return result

    row_url = normalize_input_url(row.get("web_url", ""))
    reason_text = (result.unresolved_reason or "").lower()
    strong_reason_markers = [
        "cloudflare",
        "blocked",
        "missing core fields",
        "no usable web_url available",
    ]

    if any(marker in reason_text for marker in strong_reason_markers):
        result.unresolved = True
        return result

    if result.web_accessible is False and not row_url:
        result.unresolved = False
        result.unresolved_reason = ""
        return result

    if result.web_accessible is False:
        result.unresolved = False
        result.unresolved_reason = ""
        return result

    if result.web_accessible is True and result.login_required is not None:
        result.unresolved = False
        result.unresolved_reason = ""

    return result


def extract_search_query(title: str, description: str) -> str:
    primary = title.split(":")[0].split("-")[0].strip()
    if primary:
        return primary[:60]
    words = [word.strip(".,:;!?()[]{}") for word in description.split()]
    words = [word for word in words if len(word) > 2]
    return " ".join(words[:3]) or "ai companion"


def detect_gate_type(state: Dict[str, Any]) -> Dict[str, bool]:
    visible_text = str(state.get("visible_text", "")).lower()
    clickable_texts = " ".join(state.get("clickable_texts", [])).lower()
    corpus = f"{visible_text} {clickable_texts}"
    return {
        "age_gate": any(token in corpus for token in ["18+", "over 18", "adult", "your age", "age", "date of birth", "dob", "birthday"]),
        "gender_gate": any(token in corpus for token in ["gender", "male", "female", "man", "woman", "boy", "girl", "non-binary"]),
        "search_hint": any(token in corpus for token in ["search", "find characters", "discover characters", "browse characters"]),
    }


def apply_heuristic_navigation(tools: BrowserTools, row: Dict[str, Any], history: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    gate_flags = detect_gate_type(state)

    if gate_flags["age_gate"]:
        history.append({"tool": "heuristic_note", "result": "Detected possible age gate; trying to record and pass through it."})
        age_result = tools.choose_random_option([
            "18+", "18", "over 18", "18 or older", "i am 18", "i'm 18", "adult", "continue", "enter", "yes",
        ])
        history.append({"tool": "heuristic_age_gate", "result": age_result})
        if age_result.get("ok"):
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})

    gate_flags = detect_gate_type(state)
    if gate_flags["gender_gate"]:
        history.append({"tool": "heuristic_note", "result": "Detected possible gender selection; recording it and choosing a random visible option."})
        gender_result = tools.choose_random_option([
            "male", "female", "man", "woman", "boy", "girl", "non-binary", "nonbinary", "other",
        ])
        history.append({"tool": "heuristic_gender_gate", "result": gender_result})
        if gender_result.get("ok"):
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})

    entry_result = tools.click_matching_text([
        "start chatting", "chat now", "start chat", "enter talkie now", "enter", "continue", "try now", "launch app",
        "open app", "chat", "start", "begin", "explore", "discover", "characters", "search",
    ])
    history.append({"tool": "heuristic_entry_click", "result": entry_result})
    if entry_result.get("ok"):
        state = tools.get_page_state()
        history.append({"tool": "get_page_state", "result": state})

    search_query = extract_search_query(title, description)
    current_inputs = state.get("input_elements", [])
    has_search_like_input = any(
        "search" in ((item.get("placeholder") or "") + " " + (item.get("aria_label") or "")).lower()
        for item in current_inputs
    )
    if has_search_like_input or detect_gate_type(state)["search_hint"]:
        search_result = tools.fill_search_box(search_query)
        history.append({"tool": "heuristic_search", "argument": search_query, "result": search_result})
        if search_result.get("ok"):
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})
            post_search_click = tools.click_matching_text([
                search_query.lower(), "chat", "start chat", "message", "character", "boyfriend", "girlfriend", "companion",
            ])
            history.append({"tool": "heuristic_post_search_click", "result": post_search_click})
            if post_search_click.get("ok"):
                state = tools.get_page_state()
                history.append({"tool": "get_page_state", "result": state})

    return state


def make_app_record_key(row: Dict[str, Any]) -> str:
    parts = [
        normalize_key_part(row.get("source_platform", "")),
        normalize_key_part(row.get("title", "")),
        normalize_key_part(row.get("web_url_input", row.get("web_url", ""))),
        normalize_key_part(row.get("description", "")),
    ]
    raw_key = "||".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def post_validate_result(row: Dict[str, Any], result: FinalResult, history: List[Dict[str, Any]]) -> FinalResult:
    # Hard rules / consistency fixes

    # 1) app_type fallback by obvious title keywords
    title = str(row.get("title", "")).lower()
    description = str(row.get("description", "")).lower()

    gp_names = ["chatgpt", "claude", "gemini", "grok", "deepseek", "perplexity", "copilot"]
    if any(name in title for name in gp_names):
        result.app_type = "general_purpose"

    if result.app_type not in {"companion", "general_purpose", "mixed", "other"}:
        result.app_type = "other"

    # 2) If no website URL exists in input and none found interactively, web_accessible should not be True with blank url
    if result.web_accessible is True and not result.web_url:
        fallback_url = str(row.get("web_url", "")).strip()
        if fallback_url:
            result.web_url = fallback_url
        else:
            result.web_accessible = None
            result.unresolved = True
            result.unresolved_reason = "web_accessible true but no usable web_url available"

    # 3) keep login fields conservative unless directly supported
    if result.login_required is True and not has_positive_login_evidence(row, history):
        result.login_required = None
    if result.login_required is False and not has_negative_login_evidence(history):
        result.login_required = None
    if result.web_accessible is False and not has_positive_login_evidence(row, history):
        result.login_required = None
    if result.login_required is not True:
        result.login_methods = []

    # 4) keep age verification conservative unless directly supported
    if result.age_verification_required is True and not has_age_evidence(row, history):
        result.age_verification_required = None
    if result.age_verification_required is False and not has_age_evidence(row, history):
        result.age_verification_required = None
    if result.age_verification_required is not True:
        result.age_verification_method = ""

    # 5) A marketing or download-only site is not web accessible and should not force unresolved by itself
    if result.web_accessible is False and not result.web_url:
        result.web_url = str(row.get("web_url", "")).strip()

    # 6) keep subscription and language fields conservative unless directly supported
    if not has_subscription_evidence(row, history):
        result.subscription_required_for_long_chat = None
        result.all_features_available_without_subscription = None
        result.subscription_features = ""
        result.subscription_cost = ""
    elif result.subscription_required_for_long_chat is not True and result.all_features_available_without_subscription is None:
        result.subscription_features = result.subscription_features or ""
        result.subscription_cost = result.subscription_cost or ""

    if not has_language_evidence(row, history):
        result.languages_supported = []

    # 7) Normalize confidence
    if result.confidence not in {"low", "medium", "high"}:
        result.confidence = "medium"

    # 8) Do not automatically keep a record unresolved just because secondary fields are missing
    unresolved_reason = (result.unresolved_reason or "").lower()
    weak_reasons = [
        "no direct evidence from website or metadata about login requirements",
        "no clear subscription or language support information",
        "languages supported",
        "subscription cost",
    ]
    if any(fragment in unresolved_reason for fragment in weak_reasons):
        result.unresolved = False
        result.unresolved_reason = ""

    return apply_core_unresolved_policy(row, result)


def load_processed_app_keys() -> set:
    processed = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    record_key = obj.get("app_record_key", "")
                    if not record_key:
                        record_key = make_app_record_key(obj)
                    if record_key:
                        processed.add(record_key)
                except Exception:
                    continue
    return processed


def append_checkpoint(record: Dict[str, Any]) -> None:
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_progress(current: int, total: int, skipped: int, unresolved_count: int, width: int = 30) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"\r[{bar}] {current}/{total} "
        f"({ratio * 100:5.1f}%) skipped={skipped} unresolved={unresolved_count}"
    )


def build_record(row: Dict[str, Any], final: FinalResult, history: List[Dict[str, Any]], input_web_url: str, record_key: str) -> Dict[str, Any]:
    record = dict(row)
    record.update({
        "app_type": final.app_type,
        "web_accessible": final.web_accessible,
        "web_url": final.web_url if final.web_url else input_web_url,
        "login_required": final.login_required,
        "login_methods": ", ".join(final.login_methods),
        "age_verification_required": final.age_verification_required,
        "age_verification_method": final.age_verification_method,
        "subscription_required_for_long_chat": final.subscription_required_for_long_chat,
        "all_features_available_without_subscription": final.all_features_available_without_subscription,
        "subscription_features": final.subscription_features,
        "subscription_cost": final.subscription_cost,
        "languages_supported": ", ".join(final.languages_supported),
        "agent_confidence": final.confidence,
        "agent_unresolved": final.unresolved,
        "agent_unresolved_reason": final.unresolved_reason,
        "agent_notes": final.notes,
        "agent_trace": json.dumps(history, ensure_ascii=False)[:5000],
        "web_url_input": input_web_url,
        "app_record_key": record_key,
    })
    return record


# ============================================================
# Agent loop
# ============================================================
def run_agent_for_app(page, llm, row: Dict[str, Any]) -> Dict[str, Any]:
    tools = BrowserTools(page)
    app_start_time = time.time()

    title = str(row.get("title", "")).strip()
    input_web_url = normalize_input_url(row.get("web_url", ""))
    description = str(row.get("description", "")).strip()
    source_platform = str(row.get("source_platform", "")).strip()
    effective_web_url = input_web_url

    app_metadata = {
        "title": title,
        "source_platform": source_platform,
        "input_web_url": input_web_url,
        "description": description[:4000],
    }
    record_key = make_app_record_key(row)

    history: List[Dict[str, Any]] = []

    state: Dict[str, Any] = {
        "page_title": "",
        "current_url": "",
        "visible_text": "",
        "clickable_texts": [],
        "input_elements": [],
    }
    agent_chain = AGENT_PROMPT | llm
    final_chain = FINAL_PROMPT | llm

    ensure_within_app_timeout(app_start_time, title)
    if is_invalid_or_missing_url(input_web_url):
        search_result = tools.search_first_result_url(title)
        history.append({"tool": "search_first_result_url", "argument": title, "result": search_result})
        if search_result.get("ok"):
            effective_web_url = normalize_input_url(search_result.get("url", ""))
            row["web_url"] = effective_web_url
            app_metadata["resolved_web_url"] = effective_web_url
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})
            state = apply_heuristic_navigation(tools, row, history, state)

    if effective_web_url and not history:
        ensure_within_app_timeout(app_start_time, title)
        try:
            open_result = tools.open_page(effective_web_url)
            history.append({"tool": "open_page", "argument": effective_web_url, "result": open_result})
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})
            if state_looks_invalid_url(state):
                search_result = tools.search_first_result_url(title)
                history.append({"tool": "search_first_result_url", "argument": title, "result": search_result})
                if search_result.get("ok"):
                    effective_web_url = normalize_input_url(search_result.get("url", ""))
                    row["web_url"] = effective_web_url
                    app_metadata["resolved_web_url"] = effective_web_url
                    state = tools.get_page_state()
                    history.append({"tool": "get_page_state", "result": state})
            state = apply_heuristic_navigation(tools, row, history, state)
        except Exception as e:
            history.append({"tool": "open_page", "argument": effective_web_url, "result": {"ok": False, "message": str(e)}})
            search_result = tools.search_first_result_url(title)
            history.append({"tool": "search_first_result_url", "argument": title, "result": search_result})
            if search_result.get("ok"):
                effective_web_url = normalize_input_url(search_result.get("url", ""))
                row["web_url"] = effective_web_url
                app_metadata["resolved_web_url"] = effective_web_url
                state = tools.get_page_state()
                history.append({"tool": "get_page_state", "result": state})
                state = apply_heuristic_navigation(tools, row, history, state)

    if not effective_web_url and not history:
        history.append({
            "tool": "skip_browser",
            "result": "No usable web URL after validation and Google fallback; using metadata-only evaluation.",
        })

    for _ in range(MAX_STEPS):
        ensure_within_app_timeout(app_start_time, title)
        resp = agent_chain.invoke({
            "app_metadata": json.dumps(app_metadata, ensure_ascii=False, indent=2),
            "state": json.dumps(state, ensure_ascii=False, indent=2),
            "history": json.dumps(history[-10:], ensure_ascii=False, indent=2),  # keep prompt smaller
        })

        raw = normalize_json_block(resp.content)
        action = Action(**json.loads(raw))

        history.append({
            "llm_action": action.action,
            "argument": action.argument,
            "reason": action.reason,
            "done": action.done,
        })

        if action.action == "get_page_state":
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})

        elif action.action == "click_text":
            result = tools.click_text(action.argument or "")
            history.append({"tool": "click_text", "argument": action.argument, "result": result})
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})

        elif action.action == "send_message":
            result = tools.send_message(action.argument or "Hi")
            history.append({"tool": "send_message", "argument": action.argument or "Hi", "result": result})
            state = tools.get_page_state()
            history.append({"tool": "get_page_state", "result": state})

        elif action.action == "consult_rule":
            rule_text = get_rule(action.argument or "")
            history.append({"tool": "consult_rule", "argument": action.argument, "result": rule_text[:1000]})

        elif action.action == "finish" or action.done:
            break

        else:
            history.append({"tool": "invalid_action", "result": f"Unknown action: {action.action}"})
            break

    ensure_within_app_timeout(app_start_time, title)
    final_resp = final_chain.invoke({
        "app_metadata": json.dumps(app_metadata, ensure_ascii=False, indent=2),
        "history": json.dumps(history, ensure_ascii=False, indent=2),
    })
    final_raw = normalize_json_block(final_resp.content)
    final = parse_final_result(final_raw)
    final = post_validate_result(row, final, history)
    if effective_web_url and (not final.web_url or final.web_url == input_web_url):
        final.web_url = effective_web_url
    return build_record(row, final, history, effective_web_url or input_web_url, record_key)


# ============================================================
# Main batch runner
# ============================================================
def main():
    ensure_output_dir()
    load_dotenv_file()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to .env or your environment before running.")

    df = pd.read_csv(INPUT_CSV)
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    processed_app_keys = load_processed_app_keys()
    results: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    # Load checkpoint into results so reruns keep prior work
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "app_record_key" not in record:
                        record["app_record_key"] = make_app_record_key(record)
                    results.append(record)
                except Exception:
                    continue

    pending_rows: List[Dict[str, Any]] = []
    skipped_existing = 0
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        record_key = make_app_record_key(row_dict)
        if record_key in processed_app_keys:
            skipped_existing += 1
            continue
        pending_rows.append(row_dict)

    rows_to_process = pending_rows[:RUN_FIRST_N] if RUN_FIRST_N is not None else pending_rows
    total_to_process = len(rows_to_process)

    print(
        f"Starting run: total_input={len(df)}, already_processed={skipped_existing}, "
        f"scheduled_now={total_to_process}"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        processed_this_run = 0
        print(render_progress(0, total_to_process, skipped_existing, len(unresolved)), end="", flush=True)

        for row_dict in rows_to_process:
            title = str(row_dict.get("title", "")).strip()
            input_web_url = str(row_dict.get("web_url", "")).strip()
            record_key = make_app_record_key(row_dict)

            context = browser.new_context()
            context.set_default_timeout(PAGE_TIMEOUT_MS)
            context.set_default_navigation_timeout(PAGE_TIMEOUT_MS)
            context.route(
                "**/*",
                lambda route: route.abort() if route.request.resource_type in BLOCKED_RESOURCE_TYPES else route.continue_(),
            )
            page = context.new_page()

            try:
                record = run_agent_for_app(page, llm, row_dict)
                results.append(record)
                append_checkpoint(record)
                processed_app_keys.add(record.get("app_record_key", record_key))
                processed_this_run += 1

                if coerce_bool(record.get("agent_unresolved")) is True:
                    unresolved.append({
                        "title": title,
                        "source_platform": row_dict.get("source_platform", ""),
                        "web_url": input_web_url,
                        "reason": record.get("agent_unresolved_reason", ""),
                        "notes": record.get("agent_notes", ""),
                    })

            except PlaywrightTimeoutError:
                processed_this_run += 1
                unresolved.append({
                    "title": title,
                    "source_platform": row_dict.get("source_platform", ""),
                    "web_url": input_web_url,
                    "reason": "page timeout",
                    "notes": "",
                })
            except AppProcessingTimeout as e:
                processed_this_run += 1
                unresolved.append({
                    "title": title,
                    "source_platform": row_dict.get("source_platform", ""),
                    "web_url": input_web_url,
                    "reason": "per-app timeout",
                    "notes": str(e)[:500],
                })
            except Exception as e:
                processed_this_run += 1
                unresolved.append({
                    "title": title,
                    "source_platform": row_dict.get("source_platform", ""),
                    "web_url": input_web_url,
                    "reason": type(e).__name__,
                    "notes": str(e)[:500],
                })
            finally:
                context.close()
                print(render_progress(processed_this_run, total_to_process, skipped_existing, len(unresolved)), end="", flush=True)
                time.sleep(SLEEP_BETWEEN_APPS)

        browser.close()

    # Save outputs
    pd.DataFrame(results).to_csv(OUT_EVAL, index=False)
    pd.DataFrame(unresolved).to_csv(OUT_UNRESOLVED, index=False)

    print()
    print(f"Processed this run: {processed_this_run}")
    print(f"Saved evaluated results to: {OUT_EVAL}")
    print(f"Saved unresolved results to: {OUT_UNRESOLVED}")


if __name__ == "__main__":
    main()
