import requests
import json
import time
import uuid
import hashlib
import csv
from pathlib import Path

BASE = "https://www.yollo.ai"
BOT_ID = 148463
CHAT_URL = f"{BASE}/chat/Asumi-the-Maid-148463"

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
    "Summarize this conversation in one line."
]

OUTDIR = Path("yollo_api_output_final")
OUTDIR.mkdir(exist_ok=True)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gen_finger():
    return hashlib.md5(uuid.uuid4().hex.encode()).hexdigest()


class YolloClient:
    def __init__(self):
        self.s = requests.Session()
        self.x_finger = gen_finger()
        self.token = None
        self.guest_uid = None
        self.guest_key = None
        self.session_id = None
        self.user_id = None

        self.s.headers.update({
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": BASE,
            "referer": CHAT_URL,
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            "x-finger": self.x_finger,
            "x-language": "en",
            "x-platform": "web",
            "x-version": "999.0.0",
        })

    def check(self, r, label):
        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"{label}: non-json response: {r.status_code} {r.text[:500]}")
        if r.status_code >= 400:
            raise RuntimeError(f"{label}: http {r.status_code}: {data}")
        if str(data.get("code")) != "200":
            raise RuntimeError(f"{label}: api error: {data}")
        return data

    def create_guest(self):
        r = self.s.post(f"{BASE}/api/auth/createGuest")
        data = self.check(r, "createGuest")
        self.guest_uid = data["data"]["guestUid"]
        self.guest_key = data["data"]["guestKey"]
        return data

    def login_by_guest(self):
        payload = {
            "guestUid": self.guest_uid,
            "guestKey": self.guest_key,
        }
        r = self.s.post(f"{BASE}/api/auth/loginByGuest", data=json.dumps(payload))
        data = self.check(r, "loginByGuest")
        self.token = r.headers.get("x-auth-token") or data["data"]["idToken"]
        self.s.headers["x-auth-token"] = self.token
        return data

    def get_account(self):
        r = self.s.get(f"{BASE}/api/auth/getAccount")
        data = self.check(r, "getAccount")
        self.user_id = data["data"]["id"]
        return data

    def create_session(self):
        r = self.s.post(f"{BASE}/api/msg/createSession?botId={BOT_ID}")
        data = self.check(r, "createSession")
        self.session_id = int(data["data"]["id"])
        return data

    def get_chat_setting(self):
        r = self.s.get(f"{BASE}/api/msg/getChatSettingByBotId/{BOT_ID}")
        return self.check(r, "getChatSetting")

    def get_sessions(self):
        r = self.s.post(f"{BASE}/api/msg/getSessions", data=json.dumps({"page": 1, "size": 20}))
        return self.check(r, "getSessions")

    def get_chat_messages(self):
        candidates = [
            {"sessionId": self.session_id},
            {"chatSessionId": self.session_id},
            {"sessionId": self.session_id, "page": 1, "size": 50},
            {"chatSessionId": self.session_id, "page": 1, "size": 50},
        ]
        last_err = None
        for payload in candidates:
            try:
                r = self.s.post(f"{BASE}/api/msg/getChatMessages", data=json.dumps(payload))
                data = self.check(r, f"getChatMessages {payload}")
                if isinstance(data.get("data"), list):
                    return data["data"]
            except Exception as e:
                last_err = e
        raise RuntimeError(f"getChatMessages failed: {last_err}")

    def send_message(self, msg):
        headers = dict(self.s.headers)
        headers["x-no-show-msg-handle"] = "true"
        payload = {"sessionId": self.session_id, "msg": msg}
        r = self.s.post(f"{BASE}/api/msg/send", headers=headers, data=json.dumps(payload))
        return self.check(r, "send")

    def build_conversation_history(self, messages):
        """
        Convert Yollo message list to chat-stream conversationHistory.
        Keep only text messages (type == 0).
        role:
          assistant for bot (sendUserType == 1)
          user for human (sendUserType == 0)
        """
        history = []
        for m in messages:
            if m.get("type") != 0:
                continue
            role = "assistant" if m.get("sendUserType") == 1 else "user"
            history.append({
                "role": role,
                "content": m.get("content", "")
            })
        return history

    def stream_reply(self, current_user_message, conversation_history):
        """
        Real request body observed in requests.json:
        {
          "message": "...",
          "sessionId": "4384563",
          "conversationHistory": [...],
          "userToken": "...",
          "userLocale": "en",
          "isRegenerate": false,
          "isSafeMode": false
        }
        """
        payload = {
            "message": current_user_message,
            "sessionId": str(self.session_id),   # observed as string in captured request
            "conversationHistory": conversation_history,
            "userToken": self.token,
            "userLocale": "en",
            "isRegenerate": False,
            "isSafeMode": False,
        }

        r = self.s.post(
            f"{BASE}/chat-stream",
            headers={"accept": "text/event-stream", "content-type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False),
            stream=True,
            timeout=60,
        )

        if r.status_code >= 400:
            raise RuntimeError(f"chat-stream http {r.status_code}: {r.text[:500]}")

        ctype = r.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            raw = r.text[:1000]
            raise RuntimeError(f"chat-stream unexpected content-type={ctype}, body={raw}")

        buffer = ""
        text_chunks = []

        for raw_line in r.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.strip()
            if not line:
                continue

            # SSE line format
            if line.startswith("data:"):
                line = line[5:].strip()

            buffer += line + "\n"

            # Some SSE implementations return raw text chunks, some return JSON
            # Try JSON parse first
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data = obj.get("data", obj)

                    # direct full message object
                    if isinstance(data, dict) and data.get("sendUserType") == 1 and data.get("type") == 0 and data.get("content"):
                        r.close()
                        return {
                            "content": data["content"],
                            "raw_stream_buffer": buffer,
                            "mode": "json_message_object"
                        }

                    # chunk-like object
                    if isinstance(data, dict):
                        if "content" in data and isinstance(data["content"], str):
                            text_chunks.append(data["content"])

            except Exception:
                # Non-JSON raw chunk
                text_chunks.append(line)

        r.close()

        merged = "".join(text_chunks).strip()
        if merged:
            return {
                "content": merged,
                "raw_stream_buffer": buffer,
                "mode": "merged_chunks"
            }

        raise RuntimeError(f"chat-stream ended with no parseable reply. Raw buffer:\n{buffer[:2000]}")


def main():
    client = YolloClient()

    bootstrap = {}
    bootstrap["create_guest"] = client.create_guest()
    bootstrap["login"] = client.login_by_guest()
    bootstrap["account"] = client.get_account()
    bootstrap["create_session"] = client.create_session()
    bootstrap["chat_setting"] = client.get_chat_setting()
    bootstrap["sessions"] = client.get_sessions()

    initial_messages = client.get_chat_messages()

    print("Authenticated guest account:")
    print(json.dumps(bootstrap["account"], ensure_ascii=False, indent=2))
    print(f"\nCreated chat session: {client.session_id}")

    bot_opening = [m for m in initial_messages if m.get("sendUserType") == 1 and m.get("type") == 0]
    if bot_opening:
        print("\nInitial opening message:")
        print(bot_opening[-1]["content"][:1200])

    save_json(OUTDIR / "bootstrap.json", bootstrap)
    save_json(OUTDIR / "initial_messages.json", initial_messages)

    results = []
    current_messages = list(initial_messages)

    for idx, user_msg in enumerate(MESSAGES, start=1):
        print(f"\n--- TURN {idx} ---")
        print("USER:", user_msg)

        t0 = time.perf_counter()

        send_result = client.send_message(user_msg)
        user_message_id = send_result["data"]["id"]

        # update local history with the just-sent user msg
        current_messages.append(send_result["data"])
        conversation_history = client.build_conversation_history(current_messages)

        stream_result = client.stream_reply(user_msg, conversation_history)
        latency_sec = round(time.perf_counter() - t0, 3)

        bot_reply = stream_result["content"]

        print("BOT:", bot_reply[:1200])
        print("LATENCY:", latency_sec)

        # append synthetic bot msg to local history so next turn has full context
        synthetic_bot_msg = {
            "id": None,
            "chatSessionId": client.session_id,
            "content": bot_reply,
            "sendUserType": 1,
            "receiveUserType": 0,
            "type": 0,
        }
        current_messages.append(synthetic_bot_msg)

        results.append({
            "turn_index": idx,
            "chat_session_id": client.session_id,
            "user_message_id": user_message_id,
            "user_input": user_msg,
            "bot_reply": bot_reply,
            "latency_sec": latency_sec,
            "stream_mode": stream_result["mode"],
        })

        save_json(OUTDIR / f"turn_{idx}_stream_debug.json", stream_result)
        time.sleep(0.8)

    save_json(OUTDIR / "results.json", results)
    save_csv(OUTDIR / "results.csv", results)

    print(f"\nSaved outputs to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()