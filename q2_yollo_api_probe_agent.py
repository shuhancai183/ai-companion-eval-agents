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
from pydantic import BaseModel
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


ENV_FILE = ".env"
MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = "https://api.chatanywhere.org/v1"
LLM_TIMEOUT_SECONDS = 60

TARGET_URL = "https://www.yollo.ai/chat/Asumi-the-Maid-148463"
HEADLESS = True
PAGE_TIMEOUT_MS = 20000
POST_SEND_WAIT_SECONDS = 8.0
TEST_MESSAGE = "Hello from the automated network probe."
MAX_CAPTURED_BODY_CHARS = 3000
MAX_CAPTURED_REQUESTS = 120

DEFAULT_OUTDIR = Path("yollo_api_probe_output")


class ApiPlan(BaseModel):
    site_name: str
    base_url: str
    chat_url: str
    create_guest_url: Optional[str] = None
    login_by_guest_url: Optional[str] = None
    get_account_url: Optional[str] = None
    create_session_url: Optional[str] = None
    get_chat_setting_url: Optional[str] = None
    get_sessions_url: Optional[str] = None
    get_chat_messages_url: Optional[str] = None
    send_url: Optional[str] = None
    stream_url: Optional[str] = None
    bot_id: Optional[str] = None
    required_headers: Dict[str, str] = {}
    payload_hints: Dict[str, Dict[str, Any]] = {}
    conversation_history_format: Optional[str] = None
    token_location: Optional[str] = None
    session_id_location: Optional[str] = None
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


def coerce_string_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    cleaned: Dict[str, str] = {}
    for key, item in value.items():
        key_s = coerce_str(key)
        item_s = coerce_str(item)
        if key_s and item_s:
            cleaned[key_s] = item_s
    return cleaned


def coerce_payload_hints(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    cleaned: Dict[str, Dict[str, Any]] = {}
    for key, item in value.items():
        key_s = coerce_str(key)
        if not key_s:
            continue
        if isinstance(item, dict):
            cleaned[key_s] = item
        elif isinstance(item, list):
            cleaned[key_s] = {str(field): None for field in item}
        elif item is None:
            cleaned[key_s] = {}
        else:
            cleaned[key_s] = {"value": item}
    return cleaned


def flatten_required_headers(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    flat = {}
    for key, item in value.items():
        if isinstance(item, dict):
            flat.update(coerce_string_dict(item))
        else:
            key_s = coerce_str(key)
            item_s = coerce_str(item)
            if key_s and item_s:
                flat[key_s] = item_s
    return flat


def site_key_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def normalize_api_plan_payload(payload: Dict[str, Any], default_chat_url: str) -> Dict[str, Any]:
    normalized = dict(payload)

    for key in [
        "site_name",
        "base_url",
        "chat_url",
        "create_guest_url",
        "login_by_guest_url",
        "get_account_url",
        "create_session_url",
        "get_chat_setting_url",
        "get_sessions_url",
        "get_chat_messages_url",
        "send_url",
        "stream_url",
        "bot_id",
        "token_location",
        "session_id_location",
        "notes",
    ]:
        normalized[key] = coerce_str(normalized.get(key))

    normalized["required_headers"] = flatten_required_headers(normalized.get("required_headers"))
    normalized["payload_hints"] = coerce_payload_hints(normalized.get("payload_hints"))

    conversation_history_format = normalized.get("conversation_history_format")
    if isinstance(conversation_history_format, (dict, list)):
        normalized["conversation_history_format"] = json.dumps(conversation_history_format, ensure_ascii=False)
    else:
        normalized["conversation_history_format"] = coerce_str(conversation_history_format)

    if not normalized.get("site_name"):
        normalized["site_name"] = site_key_from_url(default_chat_url)
    if not normalized.get("base_url"):
        parsed = urlparse(default_chat_url)
        normalized["base_url"] = f"{parsed.scheme}://{parsed.netloc}"
    if not normalized.get("chat_url"):
        normalized["chat_url"] = default_chat_url
    if not normalized.get("notes"):
        normalized["notes"] = ""

    return normalized


def clean_headers(headers: Dict[str, str]) -> Dict[str, str]:
    allowed = {
        "accept",
        "accept-language",
        "content-type",
        "origin",
        "referer",
        "user-agent",
        "x-auth-token",
        "x-finger",
        "x-language",
        "x-no-show-msg-handle",
        "x-platform",
        "x-version",
    }
    cleaned: Dict[str, str] = {}
    for key, value in headers.items():
        key_l = key.lower()
        if key_l in allowed and value:
            cleaned[key_l] = value
    return cleaned


def try_parse_json(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def endpoint_key(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path


def looks_like_api(url: str, target_site_key: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    same_site = host == target_site_key or host.endswith("." + target_site_key) or target_site_key.endswith("." + host)
    return same_site and (
        "/api/" in parsed.path or
        parsed.path.endswith("/chat-stream") or
        parsed.path.endswith("/stream") or
        parsed.path.endswith("/messages") or
        parsed.path.endswith("/chat")
    )


def extract_bot_id(url: str) -> Optional[str]:
    match = re.search(r"-(\d+)(?:$|[/?#])", url)
    if match:
        return match.group(1)
    return None


def summarize_captures(captures: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in captures:
        by_path[item["path"]].append(item)

    summary: Dict[str, Any] = {"endpoints": []}
    for path, items in sorted(by_path.items()):
        first = items[0]
        summary["endpoints"].append({
            "path": path,
            "count": len(items),
            "method": first.get("method"),
            "sample_url": first.get("url"),
            "sample_request_headers": first.get("request_headers"),
            "sample_request_json": first.get("request_json"),
            "sample_request_body": first.get("request_body"),
            "sample_status": first.get("status"),
            "sample_response_json": first.get("response_json"),
            "sample_response_body": first.get("response_body"),
        })
    return summary


def find_request_sample(captures: List[Dict[str, Any]], url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    wanted_path = endpoint_key(url)
    for item in captures:
        if item["path"] == wanted_path:
            return item
    return None


def replace_known_fields(data: Any, replacements: Dict[str, Any]) -> Any:
    if isinstance(data, dict):
        updated = {}
        for key, value in data.items():
            if key in replacements:
                updated[key] = replacements[key]
            else:
                updated[key] = replace_known_fields(value, replacements)
        return updated
    if isinstance(data, list):
        return [replace_known_fields(item, replacements) for item in data]
    return data


def build_session_headers(base_headers: Dict[str, str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    referer = TARGET_URL
    origin = "https://www.yollo.ai"
    if extra and extra.get("__referer__"):
        referer = extra["__referer__"]
        parsed = urlparse(referer)
        origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": origin,
        "referer": referer,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "x-language": "en",
        "x-platform": "web",
        "x-version": "999.0.0",
    }
    headers.update(base_headers or {})
    if extra:
        extra = {k: v for k, v in extra.items() if not k.startswith("__")}
        headers.update(extra)
    return headers


def gen_finger() -> str:
    return hashlib.md5(uuid.uuid4().hex.encode()).hexdigest()


def sanitize_runtime_headers(headers: Optional[Dict[str, str]], token: Optional[str] = None) -> Dict[str, str]:
    cleaned = clean_headers(headers or {})
    cleaned.pop("x-auth-token", None)
    if token:
        cleaned["x-auth-token"] = token
    return cleaned


def parse_api_json(response: requests.Response, label: str) -> Dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"{label}: non-json response: {response.status_code} {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(f"{label}: http {response.status_code}: {data}")
    return data


def collect_site_network(target_url: str) -> Dict[str, Any]:
    captures: List[Dict[str, Any]] = []
    errors: List[str] = []
    target_site_key = site_key_from_url(target_url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        context.set_default_navigation_timeout(PAGE_TIMEOUT_MS)
        page = context.new_page()

        request_map: Dict[str, Dict[str, Any]] = {}

        def on_request(request) -> None:
            if len(captures) >= MAX_CAPTURED_REQUESTS:
                return
            if not looks_like_api(request.url, target_site_key):
                return
            body = request.post_data or ""
            entry = {
                "request_id": request.url + "::" + request.method + f"::{len(request_map)}",
                "url": request.url,
                "path": endpoint_key(request.url),
                "method": request.method,
                "resource_type": request.resource_type,
                "request_headers": clean_headers(request.headers),
                "request_body": body[:MAX_CAPTURED_BODY_CHARS],
                "request_json": try_parse_json(body),
            }
            request_map[entry["request_id"]] = entry

        def on_response(response) -> None:
            if len(captures) >= MAX_CAPTURED_REQUESTS:
                return
            request = response.request
            if not looks_like_api(request.url, target_site_key):
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
            try:
                text = response.text()
            except Exception:
                text = ""
            entry["response_body"] = text[:MAX_CAPTURED_BODY_CHARS]
            entry["response_json"] = try_parse_json(text)
            captures.append(entry)

        page.on("request", on_request)
        page.on("response", on_response)

        try:
            page.goto(target_url, wait_until="commit")
            page.wait_for_timeout(3500)

            input_locator = page.locator("textarea, input[type='text'], [contenteditable='true']").first
            if input_locator.count() > 0:
                try:
                    input_locator.click()
                    try:
                        input_locator.fill(TEST_MESSAGE)
                    except Exception:
                        input_locator.type(TEST_MESSAGE, delay=20)

                    send_candidates = page.locator(
                        "button:has-text('Send'), button[aria-label*='send' i], button[type='submit']"
                    )
                    if send_candidates.count() > 0:
                        send_candidates.first.click()
                    else:
                        input_locator.press("Enter")
                except Exception as exc:
                    errors.append(f"message_send_failed: {exc}")
            else:
                errors.append("no_message_input_found")

            page.wait_for_timeout(int(POST_SEND_WAIT_SECONDS * 1000))
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
        "captures": captures,
        "summary": summarize_captures(captures),
    }


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are analyzing network captures from a web AI companion app.
Infer the minimum API calling pattern required to bootstrap a guest session, load chat state, send one message, and receive one streamed reply.

Return JSON only with these keys:
site_name, base_url, chat_url, create_guest_url, login_by_guest_url, get_account_url,
create_session_url, get_chat_setting_url, get_sessions_url, get_chat_messages_url,
send_url, stream_url, bot_id, required_headers, payload_hints, conversation_history_format,
token_location, session_id_location, notes.

Rules:
- Prefer exact URLs observed in the captures.
- Use null when unknown.
- required_headers should include only headers that appear necessary.
- payload_hints should only include keys you have evidence for from the captures.
- notes should explain uncertainty briefly.
""",
        ),
        (
            "human",
            """Target URL: {target_url}
Current URL: {current_url}
Page title: {page_title}
Bot id guess from URL: {bot_id_guess}

Captured API summary:
{capture_summary}

If there was a failure on a previous verification attempt, use it to revise the plan:
{verification_feedback}
""",
        ),
    ]
)


def derive_plan(
    llm: ChatOpenAI,
    target_url: str,
    current_url: str,
    page_title: str,
    bot_id_guess: Optional[str],
    summary: Dict[str, Any],
    verification_feedback: str = "none yet",
) -> ApiPlan:
    response = llm.invoke(
        PLAN_PROMPT.format_messages(
            target_url=target_url,
            current_url=current_url,
            page_title=page_title,
            bot_id_guess=bot_id_guess or "",
            capture_summary=json.dumps(summary, ensure_ascii=False, indent=2),
            verification_feedback=verification_feedback,
        )
    )
    raw_payload = json.loads(normalize_json_block(response.content))
    payload = normalize_api_plan_payload(raw_payload, target_url)
    return ApiPlan(**payload)


def verify_plan(plan: ApiPlan, captures: List[Dict[str, Any]]) -> Dict[str, Any]:
    session = requests.Session()
    runtime_headers = dict(plan.required_headers or {})
    runtime_headers["x-finger"] = gen_finger()
    runtime_headers.pop("x-auth-token", None)
    base_headers = build_session_headers(runtime_headers, extra={"__referer__": plan.chat_url})
    session.headers.update(base_headers)

    results: Dict[str, Any] = {
        "ok": False,
        "steps": [],
        "errors": [],
    }

    def step_request(label: str, method: str, url: str, headers: Optional[Dict[str, str]] = None, json_body: Any = None):
        merged_headers = dict(session.headers)
        if headers:
            merged_headers.update(headers)
        start = time.perf_counter()
        request_kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": merged_headers,
            "timeout": 60 if label == "chat_stream" else 30,
            "stream": (label == "chat_stream"),
        }
        if json_body is not None:
            request_kwargs["data"] = json.dumps(json_body, ensure_ascii=False)
        response = session.request(**request_kwargs)
        latency = round(time.perf_counter() - start, 3)
        item = {
            "label": label,
            "method": method,
            "url": url,
            "status_code": response.status_code,
            "latency_sec": latency,
        }
        return response, item

    try:
        create_guest_sample = find_request_sample(captures, plan.create_guest_url)
        login_sample = find_request_sample(captures, plan.login_by_guest_url)
        create_session_sample = find_request_sample(captures, plan.create_session_url)
        get_messages_sample = find_request_sample(captures, plan.get_chat_messages_url)
        send_sample = find_request_sample(captures, plan.send_url)
        stream_sample = find_request_sample(captures, plan.stream_url)

        response, item = step_request("create_guest", "POST", plan.create_guest_url)
        data = parse_api_json(response, "create_guest")
        item["response_json"] = data
        results["steps"].append(item)
        guest_uid = data["data"]["guestUid"]
        guest_key = data["data"]["guestKey"]

        login_body = (login_sample or {}).get("request_json") or {"guestUid": "", "guestKey": ""}
        login_body = replace_known_fields(login_body, {"guestUid": guest_uid, "guestKey": guest_key})
        response, item = step_request("login_by_guest", "POST", plan.login_by_guest_url, json_body=login_body)
        data = parse_api_json(response, "login_by_guest")
        item["response_json"] = data
        results["steps"].append(item)

        token = response.headers.get("x-auth-token") or data.get("data", {}).get("idToken")
        if token:
            session.headers["x-auth-token"] = token

        response, item = step_request("get_account", "GET", plan.get_account_url)
        data = parse_api_json(response, "get_account")
        item["response_json"] = data
        results["steps"].append(item)

        bot_id = plan.bot_id or extract_bot_id(plan.chat_url)

        response, item = step_request("create_session", "POST", plan.create_session_url)
        data = parse_api_json(response, "create_session")
        item["response_json"] = data
        results["steps"].append(item)
        session_id = data["data"]["id"]

        if plan.get_chat_setting_url:
            response, item = step_request("get_chat_setting", "GET", plan.get_chat_setting_url)
            item["response_json"] = parse_api_json(response, "get_chat_setting")
            results["steps"].append(item)

        if plan.get_sessions_url:
            sessions_body = ((find_request_sample(captures, plan.get_sessions_url) or {}).get("request_json") or {"page": 1, "size": 20})
            response, item = step_request("get_sessions", "POST", plan.get_sessions_url, json_body=sessions_body)
            item["response_json"] = parse_api_json(response, "get_sessions")
            results["steps"].append(item)

        current_messages: List[Dict[str, Any]] = []
        if plan.get_chat_messages_url:
            messages_body = (get_messages_sample or {}).get("request_json") or {"sessionId": session_id}
            messages_body = replace_known_fields(messages_body, {"sessionId": session_id, "chatSessionId": session_id})
            response, item = step_request("get_chat_messages", "POST", plan.get_chat_messages_url, json_body=messages_body)
            messages_json = parse_api_json(response, "get_chat_messages")
            item["response_json"] = messages_json
            results["steps"].append(item)
            if isinstance(messages_json.get("data"), list):
                current_messages = messages_json["data"]

        send_headers = sanitize_runtime_headers((send_sample or {}).get("request_headers") or {}, token=token)
        send_headers["x-no-show-msg-handle"] = "true"
        send_body = {"sessionId": session_id, "msg": TEST_MESSAGE}
        response, item = step_request("send", "POST", plan.send_url, headers=send_headers, json_body=send_body)
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
            conversation_history.append({
                "role": role,
                "content": message.get("content", ""),
            })

        stream_headers = sanitize_runtime_headers((stream_sample or {}).get("request_headers") or {}, token=token)
        stream_headers["accept"] = "text/event-stream"
        stream_headers["content-type"] = "application/json"

        stream_body = {
            "message": TEST_MESSAGE,
            "sessionId": str(session_id),
            "conversationHistory": conversation_history,
            "userToken": token,
            "userLocale": "en",
            "isRegenerate": False,
            "isSafeMode": False,
        }

        response, item = step_request("chat_stream", "POST", plan.stream_url, headers=stream_headers, json_body=stream_body)
        stream_chunks: List[str] = []
        raw_buffer = ""
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
                    stream_chunks.append(data["content"])
            except Exception:
                stream_chunks.append(line)
        item["stream_preview"] = raw_buffer[:MAX_CAPTURED_BODY_CHARS]
        item["parsed_reply"] = "".join(stream_chunks).strip()
        results["steps"].append(item)

        results["ok"] = str(send_json.get("code")) == "200" and bool(item["parsed_reply"])
        if not results["ok"]:
            results["errors"].append("verification did not achieve both a successful send step and a parseable streamed reply")
    except Exception as exc:
        results["errors"].append(str(exc))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop API probe agent for Yollo-like companion sites.")
    parser.add_argument("--target-url", default=TARGET_URL, help="Interactive chat page URL to probe.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="Directory for probe artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_url = args.target_url
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    load_dotenv_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to .env or your environment before running.")

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    capture = collect_site_network(target_url)
    save_json(outdir / "network_capture.json", capture)

    bot_id_guess = extract_bot_id(capture["current_url"] or target_url)

    plan1 = derive_plan(
        llm=llm,
        target_url=target_url,
        current_url=capture["current_url"],
        page_title=capture["page_title"],
        bot_id_guess=bot_id_guess,
        summary=capture["summary"],
    )
    save_json(outdir / "api_plan_attempt_1.json", plan1.model_dump())

    verify1 = verify_plan(plan1, capture["captures"])
    save_json(outdir / "verification_attempt_1.json", verify1)

    final_payload = {
        "target_url": target_url,
        "network_capture_file": str((outdir / "network_capture.json").resolve()),
        "plan_attempt_1_file": str((outdir / "api_plan_attempt_1.json").resolve()),
        "verification_attempt_1_file": str((outdir / "verification_attempt_1.json").resolve()),
        "final_status": "success" if verify1.get("ok") else "needs_revision",
        "success_criteria": "A run only counts as success if the send step returns API code 200 and chat_stream yields a parseable assistant reply.",
    }

    if verify1.get("ok"):
        final_payload["winning_plan"] = plan1.model_dump()
        final_payload["winning_verification"] = verify1
        save_json(outdir / "final_result.json", final_payload)
        print(f"Closed-loop probe succeeded. Outputs saved to: {outdir.resolve()}")
        return

    feedback = json.dumps(verify1, ensure_ascii=False, indent=2)
    plan2 = derive_plan(
        llm=llm,
        target_url=target_url,
        current_url=capture["current_url"],
        page_title=capture["page_title"],
        bot_id_guess=bot_id_guess,
        summary=capture["summary"],
        verification_feedback=feedback,
    )
    save_json(outdir / "api_plan_attempt_2.json", plan2.model_dump())

    verify2 = verify_plan(plan2, capture["captures"])
    save_json(outdir / "verification_attempt_2.json", verify2)

    final_payload["plan_attempt_2_file"] = str((outdir / "api_plan_attempt_2.json").resolve())
    final_payload["verification_attempt_2_file"] = str((outdir / "verification_attempt_2.json").resolve())
    final_payload["final_status"] = "success" if verify2.get("ok") else "failed_after_two_attempts"
    final_payload["winning_plan"] = plan2.model_dump() if verify2.get("ok") else plan1.model_dump()
    final_payload["winning_verification"] = verify2 if verify2.get("ok") else verify1

    save_json(outdir / "final_result.json", final_payload)
    print(f"Closed-loop probe finished. Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
