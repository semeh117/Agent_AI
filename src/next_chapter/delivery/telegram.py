"""
send_results_telegram.py
--------------
Delivers the candidate's job matches + cover letter via a Telegram bot
message — the Telegram counterpart of send_results_email.py (which creates
a Gmail DRAFT). Both produce the SAME content; the agent picks the channel
by asking the user.

Lands as a message in a Telegram chat, not a draft: unlike Gmail's
gmail.compose scope, the Bot API has no "draft" concept — you send a
message and it's delivered. That's the intended default here too: the
recipient reads it in their chat and can forward/store it as they like.

Requires two env values (see .env.example):
    TELEGRAM_BOT_TOKEN   — token from @BotFather
    TELEGRAM_CHAT_ID     — numeric chat id of the recipient. Get it via
                           the `getUpdates` endpoint after messaging the
                           bot once (or your personal chat id via
                           @userinfobot). This is the candidate's own chat
                           id, same self-addressed idea as the Gmail draft.

Uses plain `requests` — no extra dependency added.
"""

import os
import hashlib

import requests

from next_chapter.delivery.gmail import _build_email_body

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def _telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise FileNotFoundError(
            "TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not set in .env. "
            "Add both — token from @BotFather, chat id from @userinfobot."
        )
    return token, chat_id


def _split_long_message(text: str, limit: int = 4000) -> list[str]:
    """
    Telegram messages are capped at 4096 chars. Split a long body on
    paragraph boundaries without ever producing an empty chunk.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            # a single paragraph longer than the limit: hard-split it
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def create_results_telegram(cv_info, ranked_jobs: list, cover_letter: str, *,
                            delivery_id: str | None = None, database_path=None) -> dict:
    """
    Sends the ranked matches + cover letter to the recipient's Telegram
    chat. Same ranked_jobs payload shape as create_results_draft().

    Args:
        cv_info: the candidate's CVInfo object (for the greeting line).
        ranked_jobs: results from run_job_matching() — already sorted.
        cover_letter: output of generate_cover_letter() for the top job.

    Returns:
        A dict with the Bot API `sendMessage` responses (one per rendered
        chunk), so the caller can confirm delivery.
    """
    token, chat_id = _telegram_credentials()

    body = _build_email_body(cv_info, ranked_jobs, cover_letter)
    api = TELEGRAM_API.format(token=token)
    from next_chapter.services.delivery_journal import claim_part, finish_part, UncertainDeliveryError
    # Include destination and content so legacy callers also get retry protection.
    content_hash = hashlib.sha256(f"{chat_id}\n{body}".encode()).hexdigest()
    operation = f"telegram:{delivery_id or content_hash}"

    responses = []
    for index, chunk in enumerate(_split_long_message(body)):
        receipt = claim_part(operation, index, content_hash, database_path=database_path)
        if receipt is not None:
            responses.append(receipt)
            continue
        try:
            response = requests.post(
                f"{api}/sendMessage",
                json={"chat_id": chat_id, "text": chunk},
                timeout=30,
            )
        except requests.RequestException:
            # Do not expose the request URL: Telegram embeds the secret in it.
            raise UncertainDeliveryError(
                f"Telegram part {index + 1} has an uncertain network outcome. "
                "Check your chat before retrying."
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise UncertainDeliveryError("Telegram returned an unreadable response; check your chat.") from None
        if response.status_code != 200 or not payload.get("ok"):
            if response.status_code >= 500:
                raise UncertainDeliveryError("Telegram reported a server error; check your chat before retrying.")
            finish_part(operation, index, None, database_path=database_path)
            try:
                detail = payload["description"]
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(
                f"Telegram sendMessage failed (HTTP {response.status_code}): {detail}"
            )
        finish_part(operation, index, payload, database_path=database_path)
        responses.append(payload)

    print(f"Telegram message(s) sent to chat {chat_id} ({len(responses)} chunk(s)).")
    return {"chat_id": chat_id, "messages": responses}
