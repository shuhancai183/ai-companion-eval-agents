import argparse
import json
import os
import re
import time
import uuid
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


ENV_FILE = ".env"
OPENAI_BASE_URL = "https://api.chatanywhere.org/v1"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL_PROBE_V2", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
DEFAULT_REVISE_MODEL = os.getenv("OPENAI_REVISE_MODEL_PROBE_V2", DEFAULT_MODEL)
LLM_TIMEOUT_SECONDS = 90

HEADLESS = True
PAGE_TIMEOUT_MS = 20000
POST_ACTION_WAIT_MS = 2500
POST_SEND_WAIT_MS = 6000
MAX_CAPTURED_BODY_CHARS = 4000
MAX_CAPTURED_REQUESTS = 250
TEST_MESSAGE = "Hello from the automated network probe."
DEFAULT_OUTDIR = Path("site_probe_output_v2")

ENTRY_TEXT_HINTS = [
    "chat",
    "start",
    "continue",
    "enter",
    "try now",
    "message",
    "talk",
    "begin",
    "next",
    "open",
]

MESSAGE_INPUT_SELECTORS = [
    "textarea",
    "input[type='text']",
    "input:not([type])",
    "[contenteditable='true']",
]


class ApiPlanV2(BaseModel):
    site_name: str
    base_url: str
    chat_url: str
    transport: Optional[str] = None
    create_guest_url: Optional[str] = None
    login_by_guest_url: Optional[str] = None
    get_account_url: Optional[str] = None
    create_session_url: Optional[str] = None
    get_chat_setting_url: Optional[str] = None
    get_sessions_url: Optional[str] = None
    get_chat_messages_url: Optional[str] = None
    send_url: Optional[str] = None
    stream_url: Optional[str] = None
    websocket_url: Optional[str] = None
    graphql_url: Optional[str] = None
    bot_id: Optional[str] = None
    required_headers: Dict[str, str] = {}
    payload_hints: Dict[str, Dict[str, Any]] = {}
    conversation_history_format: Optional[str] = None
    token_location: Optional[str] = None
    session_id_location: Optional[str] = None
    verification_strategy: Optional[str] = None
    confidence: Optional[str] = None
    notes: str = ""


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def site_key_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def host_matches_target(host: str, target_site_key: str) -> bool:
    host = host.lower()
    return host == target_site_key or host.endswith("." + target_site_key) or target_site_key.endswith("." + host)


def is_candidate_http_endpoint(url: str, target_site_key: str, method: str, resource_type: str, headers: Dict[str, str]) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    header_keys = {k.lower(): v for k, v in headers.items()}
    content_type = (header_keys.get("content-type", "") or "").lower()
    accept = (header_keys.get("accept", "") or "").lower()

    if host_matches_target(host, target_site_key):
        if "/api/" in path:
            return True
        if any(token in path for token in ["chat", "message", "conversation", "session", "stream", "graphql", "auth"]):
            return True
        if resource_type in {"xhr", "fetch"}:
            return True
    if "json" in content_type or "json" in accept:
        return True
    if "event-stream" in accept or "event-stream" in content_type:
        return True
    if method.upper() == "POST" and resource_type in {"xhr", "fetch"}:
        return True
    return False


def clean_headers(headers: Dict[str, str]) -> Dict[str, str]:
    allowed = {
        "accept",
        "accept-language",
        "authorization",
        "content-type",
        "cookie",
        "origin",
        "referer",
        "user-agent",
        "x-api-key",
        "x-auth-token",
        "x-client-id",
        "x-device-id",
        "x-finger",
        "x-language",
        "x-no-show-msg-handle",
        "x-platform",
        "x-version",
    }
    cleaned: Dict[str, str] = {}
    for key, value in headers.items():
        key_l = str(key).lower()
        if key_l in allowed and value:
            cleaned[key_l] = str(value)
    return cleaned


def try_parse_json(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_request_body(request) -> str:
    try:
        data = request.post_data
        return data or ""
    except Exception:
        return ""


def safe_response_text(response) -> str:
    try:
        return response.text() or ""
    except Exception:
        return ""


def endpoint_key(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def extract_bot_id(url: str) -> Optional[str]:
    match = re.search(r"-(\d+)(?:$|[/?#])", url)
    if match:
        return match.group(1)
    return None


def guess_http_transport(summary: Dict[str, Any]) -> str:
    if summary.get("websocket_connections"):
        return "websocket_or_mixed"
    for endpoint in summary.get("http_endpoints", []):
        path = (endpoint.get("path") or "").lower()
        if "graphql" in path:
            return "graphql"
        if "stream" in path:
            return "http_stream"
    return "http_json"


def coerce_string_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for key, item in value.items():
        key_s = coerce_str(key)
        if not key_s:
            continue
        if isinstance(item, dict):
            continue
        item_s = coerce_str(item)
        if item_s:
            out[key_s] = item_s
    return out


def flatten_required_headers(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out.update(coerce_string_dict(item))
        else:
            key_s = coerce_str(key)
            item_s = coerce_str(item)
            if key_s and item_s:
                out[key_s] = item_s
    return out


def coerce_payload_hints(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, item in value.items():
        key_s = coerce_str(key)
        if not key_s:
            continue
        if isinstance(item, dict):
            out[key_s] = item
        elif isinstance(item, list):
            out[key_s] = {str(field): None for field in item}
        elif item is None:
            out[key_s] = {}
        else:
            out[key_s] = {"value": item}
    return out


def normalize_api_plan_payload(payload: Dict[str, Any], target_url: str, capture_summary: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    for key in [
        "site_name",
        "base_url",
        "chat_url",
        "transport",
        "create_guest_url",
        "login_by_guest_url",
        "get_account_url",
        "create_session_url",
        "get_chat_setting_url",
        "get_sessions_url",
        "get_chat_messages_url",
        "send_url",
        "stream_url",
        "websocket_url",
        "graphql_url",
        "bot_id",
        "conversation_history_format",
        "token_location",
        "session_id_location",
        "verification_strategy",
        "confidence",
        "notes",
    ]:
        if key == "conversation_history_format" and isinstance(normalized.get(key), (dict, list)):
            normalized[key] = json.dumps(normalized.get(key), ensure_ascii=False)
        else:
            normalized[key] = coerce_str(normalized.get(key))

    normalized["required_headers"] = flatten_required_headers(normalized.get("required_headers"))
    normalized["payload_hints"] = coerce_payload_hints(normalized.get("payload_hints"))

    if not normalized.get("site_name"):
        normalized["site_name"] = site_key_from_url(target_url)
    if not normalized.get("base_url"):
        parsed = urlparse(target_url)
        normalized["base_url"] = f"{parsed.scheme}://{parsed.netloc}"
    if not normalized.get("chat_url"):
        normalized["chat_url"] = target_url
    if not normalized.get("transport"):
        normalized["transport"] = guess_http_transport(capture_summary)
    if not normalized.get("verification_strategy"):
        normalized["verification_strategy"] = "http_json" if normalized.get("send_url") and normalized.get("stream_url") else "capture_only"
    if not normalized.get("confidence"):
        normalized["confidence"] = "medium" if normalized.get("send_url") and normalized.get("stream_url") else "low"
    if not normalized.get("notes"):
        normalized["notes"] = ""
    return normalized


def summarize_captures(http_captures: List[Dict[str, Any]], ws_captures: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in http_captures:
        by_path[item["path"]].append(item)

    summary: Dict[str, Any] = {"http_endpoints": [], "websocket_connections": ws_captures[:20]}
    for path, items in sorted(by_path.items()):
        first = items[0]
        summary["http_endpoints"].append({
            "path": path,
            "count": len(items),
            "method": first.get("method"),
            "sample_url": first.get("url"),
            "resource_type": first.get("resource_type"),
            "sample_request_headers": first.get("request_headers"),
            "sample_request_json": first.get("request_json"),
            "sample_request_body": first.get("request_body"),
            "sample_status": first.get("status"),
            "sample_response_json": first.get("response_json"),
            "sample_response_body": first.get("response_body"),
        })
    return summary


def heuristic_click(page, action_log: List[Dict[str, Any]]) -> None:
    for frame in page.frames:
        for hint in ENTRY_TEXT_HINTS:
            try:
                locator = frame.locator(
                    f"button:has-text('{hint}'), a:has-text('{hint}'), [role='button']:has-text('{hint}')"
                ).first
                if locator.count() > 0:
                    locator.click(timeout=1200)
                    page.wait_for_timeout(POST_ACTION_WAIT_MS)
                    action_log.append({"action": "click_hint", "hint": hint, "frame_url": frame.url})
                    return
            except Exception:
                continue


def heuristic_send_message(page, action_log: List[Dict[str, Any]]) -> None:
    selectors = ", ".join(MESSAGE_INPUT_SELECTORS)
    for frame in page.frames:
        try:
            locator = frame.locator(selectors).first
            if locator.count() == 0:
                continue
            locator.click(timeout=1200)
            try:
                locator.fill(TEST_MESSAGE, timeout=1200)
            except Exception:
                locator.type(TEST_MESSAGE, delay=20, timeout=1200)

            send_candidates = frame.locator(
                "button:has-text('Send'), button[aria-label*='send' i], button[type='submit'], [role='button'][aria-label*='send' i]"
            )
            if send_candidates.count() > 0:
                send_candidates.first.click(timeout=1500)
            else:
                locator.press("Enter", timeout=1000)

            page.wait_for_timeout(POST_SEND_WAIT_MS)
            action_log.append({"action": "send_message", "frame_url": frame.url, "message": TEST_MESSAGE})
            return
        except Exception:
            continue
    action_log.append({"action": "send_message_failed"})


def collect_site_network_v2(target_url: str) -> Dict[str, Any]:
    http_captures: List[Dict[str, Any]] = []
    ws_captures: List[Dict[str, Any]] = []
    errors: List[str] = []
    actions: List[Dict[str, Any]] = []
    target_site_key = site_key_from_url(target_url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        context.set_default_navigation_timeout(PAGE_TIMEOUT_MS)
        page = context.new_page()

        request_map: Dict[str, Dict[str, Any]] = {}

        def on_request(request) -> None:
            if len(http_captures) >= MAX_CAPTURED_REQUESTS:
                return
            req_headers = clean_headers(request.headers)
            if not is_candidate_http_endpoint(request.url, target_site_key, request.method, request.resource_type, req_headers):
                return
            body = safe_request_body(request)
            entry = {
                "request_id": request.url + "::" + request.method + f"::{len(request_map)}",
                "url": request.url,
                "path": endpoint_key(request.url),
                "method": request.method,
                "resource_type": request.resource_type,
                "request_headers": req_headers,
                "request_body": body[:MAX_CAPTURED_BODY_CHARS],
                "request_json": try_parse_json(body),
            }
            request_map[entry["request_id"]] = entry

        def on_response(response) -> None:
            if len(http_captures) >= MAX_CAPTURED_REQUESTS:
                return
            request = response.request
            req_headers = clean_headers(request.headers)
            if not is_candidate_http_endpoint(request.url, target_site_key, request.method, request.resource_type, req_headers):
                return

            chosen_key = None
            for key, value in request_map.items():
                if value["url"] == request.url and value["method"] == request.method and "status" not in value:
                    chosen_key = key
                    break
            if chosen_key is None:
                return

            entry = request_map[chosen_key]
            entry["status"] = response.status
            entry["response_headers"] = clean_headers(response.headers)
            text = safe_response_text(response)
            entry["response_body"] = text[:MAX_CAPTURED_BODY_CHARS]
            entry["response_json"] = try_parse_json(text)
            http_captures.append(entry)

        def on_websocket(ws) -> None:
            item: Dict[str, Any] = {"url": ws.url, "frames_sent": [], "frames_received": [], "errors": []}
            ws_captures.append(item)
            ws.on("framesent", lambda payload: item["frames_sent"].append(str(payload)[:MAX_CAPTURED_BODY_CHARS]))
            ws.on("framereceived", lambda payload: item["frames_received"].append(str(payload)[:MAX_CAPTURED_BODY_CHARS]))
            ws.on("socketerror", lambda err: item["errors"].append(str(err)))

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("websocket", on_websocket)

        try:
            page.goto(target_url, wait_until="commit")
            page.wait_for_timeout(3000)
            heuristic_click(page, actions)
            heuristic_send_message(page, actions)
        except PlaywrightTimeoutError as exc:
            errors.append(f"playwright_timeout: {exc}")
        except Exception as exc:
            errors.append(f"playwright_error: {exc}")
        finally:
            current_url = page.url
            page_title = ""
            try:
                page_title = page.title()
            except Exception:
                pass
            browser.close()

    return {
        "target_url": target_url,
        "current_url": current_url,
        "page_title": page_title,
        "errors": errors,
        "actions": actions,
        "http_captures": http_captures,
        "websocket_captures": ws_captures,
        "summary": summarize_captures(http_captures, ws_captures),
    }


PLAN_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are analyzing network captures from a web AI companion app.
Your job is to infer an API calling pattern only when there is concrete evidence in the captures.

Return JSON only with these keys:
site_name, base_url, chat_url, transport, create_guest_url, login_by_guest_url, get_account_url,
create_session_url, get_chat_setting_url, get_sessions_url, get_chat_messages_url, send_url,
stream_url, websocket_url, graphql_url, bot_id, required_headers, payload_hints,
conversation_history_format, token_location, session_id_location, verification_strategy,
confidence, notes.

Rules:
- Prefer exact URLs seen in the captures.
- Do not invent endpoints just because they are common patterns.
- If send_url or stream_url were not observed, set them to null.
- If the site appears to rely on websocket traffic, say so in transport and websocket_url.
- verification_strategy should be one of: http_json, websocket_only, capture_only.
- confidence should be one of: low, medium, high.
""",
        ),
        (
            "human",
            """Target URL: {target_url}
Current URL: {current_url}
Page title: {page_title}
Observed actions:
{actions}

Captured network summary:
{capture_summary}

Previous verification feedback:
{verification_feedback}
""",
        ),
    ]
)


def derive_plan_v2(
    llm: ChatOpenAI,
    target_url: str,
    current_url: str,
    page_title: str,
    actions: List[Dict[str, Any]],
    summary: Dict[str, Any],
    verification_feedback: str = "none yet",
) -> Dict[str, Any]:
    response = llm.invoke(
        PLAN_PROMPT_V2.format_messages(
            target_url=target_url,
            current_url=current_url,
            page_title=page_title,
            actions=json.dumps(actions, ensure_ascii=False, indent=2),
            capture_summary=json.dumps(summary, ensure_ascii=False, indent=2),
            verification_feedback=verification_feedback,
        )
    )
    raw_text = response.content
    raw_payload = json.loads(normalize_json_block(raw_text))
    payload = normalize_api_plan_payload(raw_payload, target_url, summary)
    return {"raw_text": raw_text, "plan": ApiPlanV2(**payload).model_dump()}


def find_request_sample(http_captures: List[Dict[str, Any]], url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    wanted = endpoint_key(url)
    for item in http_captures:
        if item["path"] == wanted:
            return item
    return None


def build_runtime_headers(plan: Dict[str, Any]) -> Dict[str, str]:
    parsed = urlparse(plan["chat_url"])
    return {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": f"{parsed.scheme}://{parsed.netloc}",
        "referer": plan["chat_url"],
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "x-language": "en",
        "x-platform": "web",
        "x-version": "999.0.0",
        "x-finger": hashlib.md5(uuid.uuid4().hex.encode()).hexdigest(),
        **clean_headers(plan.get("required_headers", {})),
    }


def parse_api_json(response: requests.Response, label: str) -> Dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"{label}: non-json response: {response.status_code} {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(f"{label}: http {response.status_code}: {data}")
    return data


def verify_plan_v2(plan: Dict[str, Any], http_captures: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not plan.get("send_url") or not plan.get("stream_url"):
        return {
            "ok": False,
            "steps": [],
            "errors": ["Plan is incomplete for HTTP replay: send_url or stream_url is missing."],
        }

    session = requests.Session()
    session.headers.update(build_runtime_headers(plan))
    results: Dict[str, Any] = {"ok": False, "steps": [], "errors": []}

    def step_request(label: str, method: str, url: str, body: Any = None, headers: Optional[Dict[str, str]] = None, stream: bool = False):
        merged_headers = dict(session.headers)
        if headers:
            merged_headers.update(headers)
        kwargs: Dict[str, Any] = {"headers": merged_headers, "timeout": 60 if stream else 30, "stream": stream}
        if body is not None:
            kwargs["data"] = json.dumps(body, ensure_ascii=False)
        start = time.perf_counter()
        response = session.request(method, url, **kwargs)
        latency = round(time.perf_counter() - start, 3)
        return response, {"label": label, "method": method, "url": url, "status_code": response.status_code, "latency_sec": latency}

    try:
        create_guest_sample = find_request_sample(http_captures, plan.get("create_guest_url"))
        login_sample = find_request_sample(http_captures, plan.get("login_by_guest_url"))
        get_messages_sample = find_request_sample(http_captures, plan.get("get_chat_messages_url"))

        guest_uid = None
        guest_key = None
        token = None
        session_id: Optional[Any] = None
        current_messages: List[Dict[str, Any]] = []

        if plan.get("create_guest_url"):
            response, item = step_request("create_guest", "POST", plan["create_guest_url"], body=(create_guest_sample or {}).get("request_json"))
            data = parse_api_json(response, "create_guest")
            item["response_json"] = data
            results["steps"].append(item)
            guest_uid = ((data.get("data") or {}).get("guestUid") or (data.get("data") or {}).get("guest_id"))
            guest_key = ((data.get("data") or {}).get("guestKey") or (data.get("data") or {}).get("guest_token"))

        if plan.get("login_by_guest_url"):
            body = (login_sample or {}).get("request_json") or {}
            body_json = json.dumps(body, ensure_ascii=False)
            if guest_uid and "guestUid" in body_json:
                body["guestUid"] = guest_uid
            if guest_key and "guestKey" in body_json:
                body["guestKey"] = guest_key
            response, item = step_request("login_by_guest", "POST", plan["login_by_guest_url"], body=body)
            data = parse_api_json(response, "login_by_guest")
            item["response_json"] = data
            results["steps"].append(item)
            token = response.headers.get("x-auth-token") or (data.get("data") or {}).get("idToken")
            if token:
                session.headers["x-auth-token"] = token
                session.headers["authorization"] = f"Bearer {token}"

        if plan.get("get_account_url"):
            response, item = step_request("get_account", "GET", plan["get_account_url"])
            item["response_json"] = parse_api_json(response, "get_account")
            results["steps"].append(item)

        if plan.get("create_session_url"):
            response, item = step_request("create_session", "POST", plan["create_session_url"])
            data = parse_api_json(response, "create_session")
            item["response_json"] = data
            results["steps"].append(item)
            if isinstance(data.get("data"), dict):
                session_id = data["data"].get("id") or data["data"].get("sessionId")

        if plan.get("get_chat_messages_url"):
            body = (get_messages_sample or {}).get("request_json") or {}
            if session_id is not None:
                body_json = json.dumps(body, ensure_ascii=False)
                if "sessionId" in body_json:
                    body["sessionId"] = session_id
                if "chatSessionId" in body_json:
                    body["chatSessionId"] = session_id
            response, item = step_request("get_chat_messages", "POST", plan["get_chat_messages_url"], body=body or None)
            data = parse_api_json(response, "get_chat_messages")
            item["response_json"] = data
            results["steps"].append(item)
            if isinstance(data.get("data"), list):
                current_messages = data["data"]

        send_body = {"msg": TEST_MESSAGE}
        if session_id is not None:
            send_body["sessionId"] = session_id
        response, item = step_request("send", "POST", plan["send_url"], body=send_body, headers={"x-no-show-msg-handle": "true"})
        send_json = parse_api_json(response, "send")
        item["response_json"] = send_json
        results["steps"].append(item)
        if str(send_json.get("code")) != "200":
            raise RuntimeError(f"send: api error: {send_json}")
        if send_json.get("data"):
            current_messages.append(send_json["data"])

        conversation_history = []
        for message in current_messages:
            if message.get("type") != 0:
                continue
            role = "assistant" if message.get("sendUserType") == 1 else "user"
            conversation_history.append({"role": role, "content": message.get("content", "")})

        stream_body = {
            "message": TEST_MESSAGE,
            "sessionId": str(session_id) if session_id is not None else "",
            "conversationHistory": conversation_history,
            "userToken": token,
            "userLocale": "en",
            "isRegenerate": False,
            "isSafeMode": False,
        }
        response, item = step_request("chat_stream", "POST", plan["stream_url"], body=stream_body, headers={"accept": "text/event-stream"}, stream=True)
        raw_buffer = ""
        chunks: List[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            raw_buffer += line + "\n"
            try:
                obj = json.loads(line)
                data = obj.get("data", obj)
                if isinstance(data, dict) and data.get("content"):
                    chunks.append(str(data["content"]))
            except Exception:
                chunks.append(line)
        item["stream_preview"] = raw_buffer[:MAX_CAPTURED_BODY_CHARS]
        item["parsed_reply"] = "".join(chunks).strip()
        results["steps"].append(item)
        results["ok"] = bool(item["parsed_reply"])
        if not results["ok"]:
            results["errors"].append("chat_stream returned but no parseable reply was extracted")
    except Exception as exc:
        results["errors"].append(str(exc))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broader cross-site API probe agent for companion sites.")
    parser.add_argument("--target-url", required=True, help="Interactive site URL to probe.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="Directory for artifacts.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Primary model for plan extraction.")
    parser.add_argument("--revise-model", default=DEFAULT_REVISE_MODEL, help="Model for second-pass revision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    load_dotenv_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to .env or your environment before running.")

    llm = ChatOpenAI(model=args.model, temperature=0, api_key=api_key, base_url=OPENAI_BASE_URL, timeout=LLM_TIMEOUT_SECONDS)
    revise_llm = ChatOpenAI(model=args.revise_model, temperature=0, api_key=api_key, base_url=OPENAI_BASE_URL, timeout=LLM_TIMEOUT_SECONDS)

    capture = collect_site_network_v2(args.target_url)
    save_json(outdir / "network_capture.json", capture)

    plan_attempt_1 = derive_plan_v2(
        llm=llm,
        target_url=args.target_url,
        current_url=capture["current_url"],
        page_title=capture["page_title"],
        actions=capture["actions"],
        summary=capture["summary"],
    )
    save_json(outdir / "api_plan_attempt_1_raw.json", {"raw_text": plan_attempt_1["raw_text"]})
    save_json(outdir / "api_plan_attempt_1.json", plan_attempt_1["plan"])

    verify1 = verify_plan_v2(plan_attempt_1["plan"], capture["http_captures"])
    save_json(outdir / "verification_attempt_1.json", verify1)

    final_payload: Dict[str, Any] = {
        "target_url": args.target_url,
        "network_capture_file": str((outdir / "network_capture.json").resolve()),
        "plan_attempt_1_file": str((outdir / "api_plan_attempt_1.json").resolve()),
        "verification_attempt_1_file": str((outdir / "verification_attempt_1.json").resolve()),
        "final_status": "success" if verify1.get("ok") else "needs_revision",
        "success_criteria": "Success requires at least one observed calling plan with enough concrete evidence to replay a send step and obtain a parseable reply.",
    }

    if verify1.get("ok"):
        final_payload["winning_plan"] = plan_attempt_1["plan"]
        final_payload["winning_verification"] = verify1
        save_json(outdir / "final_result.json", final_payload)
        print(f"V2 probe succeeded. Outputs saved to: {outdir.resolve()}")
        return

    feedback = json.dumps(verify1, ensure_ascii=False, indent=2)
    plan_attempt_2 = derive_plan_v2(
        llm=revise_llm,
        target_url=args.target_url,
        current_url=capture["current_url"],
        page_title=capture["page_title"],
        actions=capture["actions"],
        summary=capture["summary"],
        verification_feedback=feedback,
    )
    save_json(outdir / "api_plan_attempt_2_raw.json", {"raw_text": plan_attempt_2["raw_text"]})
    save_json(outdir / "api_plan_attempt_2.json", plan_attempt_2["plan"])

    verify2 = verify_plan_v2(plan_attempt_2["plan"], capture["http_captures"])
    save_json(outdir / "verification_attempt_2.json", verify2)

    final_payload["plan_attempt_2_file"] = str((outdir / "api_plan_attempt_2.json").resolve())
    final_payload["verification_attempt_2_file"] = str((outdir / "verification_attempt_2.json").resolve())
    final_payload["final_status"] = "success" if verify2.get("ok") else "failed_after_two_attempts"
    final_payload["winning_plan"] = plan_attempt_2["plan"] if verify2.get("ok") else plan_attempt_1["plan"]
    final_payload["winning_verification"] = verify2 if verify2.get("ok") else verify1
    save_json(outdir / "final_result.json", final_payload)
    print(f"V2 probe finished. Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
