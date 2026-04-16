import argparse
import csv
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


PROBE_RESULT = Path("yollo_api_probe_output/final_result.json")
DEFAULT_OUTDIR = Path("yollo_api_probe_replay_output")

MESSAGES = [
    "Hi",
    "Can you introduce yourself in one sentence?",
    "What are you doing right now?",
    "What kind of personality do you have?",
    "Tell me something playful.",
    "What would you cook for dinner?",
    "What do you think of lazy people?",
    "Give me a short fictional scene in your style.",
    "Now ask me a question.",
    "Summarize this conversation in one line.",
]


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gen_finger() -> str:
    return hashlib.md5(uuid.uuid4().hex.encode()).hexdigest()


def clean_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
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
    for key, value in (headers or {}).items():
        key_l = str(key).lower()
        if key_l in allowed and value:
            cleaned[key_l] = str(value)
    return cleaned


def build_session_headers(plan_headers: Dict[str, str], referer: str) -> Dict[str, str]:
    parsed = requests.utils.urlparse(referer)
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
        "x-finger": gen_finger(),
    }
    for key, value in clean_headers(plan_headers).items():
        if key != "x-auth-token":
            headers[key] = value
    return headers


def parse_api_json(response: requests.Response, label: str) -> Dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"{label}: non-json response: {response.status_code} {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(f"{label}: http {response.status_code}: {data}")
    if str(data.get("code")) != "200":
        raise RuntimeError(f"{label}: api error: {data}")
    return data


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    label: str,
    headers: Optional[Dict[str, str]] = None,
    body: Any = None,
) -> Dict[str, Any]:
    merged_headers = dict(session.headers)
    if headers:
        merged_headers.update(headers)
    kwargs: Dict[str, Any] = {"headers": merged_headers, "timeout": 30}
    if body is not None:
        kwargs["data"] = json.dumps(body, ensure_ascii=False)
    response = session.request(method, url, **kwargs)
    return parse_api_json(response, label)


def build_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for message in messages:
        if message.get("type") != 0:
            continue
        role = "assistant" if message.get("sendUserType") == 1 else "user"
        history.append({
            "role": role,
            "content": message.get("content", ""),
        })
    return history


def stream_reply(
    session: requests.Session,
    stream_url: str,
    session_id: int,
    token: str,
    current_user_message: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "message": current_user_message,
        "sessionId": str(session_id),
        "conversationHistory": conversation_history,
        "userToken": token,
        "userLocale": "en",
        "isRegenerate": False,
        "isSafeMode": False,
    }

    response = session.post(
        stream_url,
        headers={"accept": "text/event-stream", "content-type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False),
        stream=True,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"chat_stream: http {response.status_code}: {response.text[:500]}")

    ctype = response.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        raise RuntimeError(f"chat_stream: unexpected content-type={ctype}, body={response.text[:1000]}")

    buffer = ""
    text_chunks: List[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()

        buffer += line + "\n"

        try:
            obj = json.loads(line)
            data = obj.get("data", obj)
            if isinstance(data, dict) and data.get("sendUserType") == 1 and data.get("type") == 0 and data.get("content"):
                response.close()
                return {
                    "content": data["content"],
                    "raw_stream_buffer": buffer,
                    "mode": "json_message_object",
                }
            if isinstance(data, dict) and isinstance(data.get("content"), str):
                text_chunks.append(data["content"])
        except Exception:
            text_chunks.append(line)

    response.close()

    merged = "".join(text_chunks).strip()
    if merged:
        return {
            "content": merged,
            "raw_stream_buffer": buffer,
            "mode": "merged_chunks",
        }

    raise RuntimeError(f"chat_stream: ended with no parseable reply. Raw buffer:\n{buffer[:2000]}")


def load_winning_plan(probe_result_path: Path) -> Dict[str, Any]:
    if not probe_result_path.exists():
        raise RuntimeError(f"Missing probe result file: {probe_result_path}")
    payload = json.loads(probe_result_path.read_text(encoding="utf-8"))
    if payload.get("final_status") != "success":
        raise RuntimeError("Probe result is not marked as success. Re-run q2_yollo_api_probe_agent.py first.")
    plan = payload.get("winning_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("final_result.json does not contain a valid winning_plan.")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a 10-turn conversation using a successful probe result.")
    parser.add_argument("--probe-result", default=str(PROBE_RESULT), help="Path to final_result.json from the probe step.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="Directory for replay artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_result = Path(args.probe_result)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plan = load_winning_plan(probe_result)

    session = requests.Session()
    session.headers.update(build_session_headers(plan.get("required_headers", {}), plan["chat_url"]))

    bootstrap: Dict[str, Any] = {}

    bootstrap["create_guest"] = request_json(session, "POST", plan["create_guest_url"], "create_guest")
    guest_uid = bootstrap["create_guest"]["data"]["guestUid"]
    guest_key = bootstrap["create_guest"]["data"]["guestKey"]

    bootstrap["login_by_guest"] = request_json(
        session,
        "POST",
        plan["login_by_guest_url"],
        "login_by_guest",
        body={"guestUid": guest_uid, "guestKey": guest_key},
    )
    token = bootstrap["login_by_guest"]["data"]["idToken"]
    session.headers["x-auth-token"] = token

    bootstrap["get_account"] = request_json(session, "GET", plan["get_account_url"], "get_account")
    bootstrap["create_session"] = request_json(session, "POST", plan["create_session_url"], "create_session")
    session_id = int(bootstrap["create_session"]["data"]["id"])

    if plan.get("get_chat_setting_url"):
        bootstrap["get_chat_setting"] = request_json(
            session, "GET", plan["get_chat_setting_url"], "get_chat_setting"
        )
    if plan.get("get_sessions_url"):
        bootstrap["get_sessions"] = request_json(
            session, "POST", plan["get_sessions_url"], "get_sessions", body={"page": 1, "size": 20}
        )

    initial_messages_resp = request_json(
        session,
        "POST",
        plan["get_chat_messages_url"],
        "get_chat_messages_initial",
        body={"chatSessionId": session_id, "size": 50},
    )
    initial_messages = initial_messages_resp["data"]

    save_json(outdir / "bootstrap.json", bootstrap)
    save_json(outdir / "initial_messages.json", initial_messages)

    current_messages = list(initial_messages)
    results: List[Dict[str, Any]] = []

    for idx, user_msg in enumerate(MESSAGES, start=1):
        print(f"\n--- TURN {idx} ---")
        print("USER:", user_msg)

        send_headers = {"x-no-show-msg-handle": "true"}
        start = time.perf_counter()
        send_result = request_json(
            session,
            "POST",
            plan["send_url"],
            "send",
            headers=send_headers,
            body={"sessionId": session_id, "msg": user_msg},
        )
        user_message = send_result["data"]
        current_messages.append(user_message)

        conversation_history = build_conversation_history(current_messages)
        stream_result = stream_reply(
            session=session,
            stream_url=plan["stream_url"],
            session_id=session_id,
            token=token,
            current_user_message=user_msg,
            conversation_history=conversation_history,
        )
        latency_sec = round(time.perf_counter() - start, 3)

        bot_reply = stream_result["content"]
        print("BOT:", bot_reply[:1200])
        print("LATENCY:", latency_sec)

        synthetic_bot_msg = {
            "id": None,
            "chatSessionId": session_id,
            "content": bot_reply,
            "sendUserType": 1,
            "receiveUserType": 0,
            "type": 0,
        }
        current_messages.append(synthetic_bot_msg)

        results.append({
            "turn_index": idx,
            "chat_session_id": session_id,
            "user_message_id": user_message["id"],
            "user_input": user_msg,
            "bot_reply": bot_reply,
            "latency_sec": latency_sec,
            "stream_mode": stream_result["mode"],
        })

        save_json(outdir / f"turn_{idx}_stream_debug.json", stream_result)
        time.sleep(0.8)

    save_json(outdir / "results.json", results)
    save_csv(outdir / "results.csv", results)

    print(f"\nSaved replay outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
