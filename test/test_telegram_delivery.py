# test/test_telegram_delivery.py
"""
test_telegram_delivery.py
--------------
Verifies the Telegram delivery channel and the user channel-choice tool:

  1. _split_long_message splits over-long bodies on paragraph boundaries
     without losing content or producing empty chunks (Telegram's 4096
     char cap).
  2. create_results_telegram builds the same body the Gmail draft uses and
     POSTs each chunk to the Bot API endpoint with the configured chat id
     (requests is mocked — no network).
  3. ask_user_delivery_channel normalizes free-form human input ("email",
     "mail", "tg", ...) to a channel and records it in module state.
  4. The send_results_telegram agent tool returns success/error strings
     like its Gmail counterpart.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.send_results_telegram import (
    _split_long_message,
    create_results_telegram,
)
from agent.tools.delivery_choice import (
    ask_user_delivery_channel,
    _last_delivery_channel,
    _CHANNELS,
)
from agent.tools.telegram_tool import send_results_telegram
import agent.tools.delivery_choice as delivery_choice
import agent.tools.telegram_tool as telegram_module


class FakeCv:
    full_name = "Jane Doe"


RANKED = [
    {
        "job_title": "AI Engineer", "company": "Acme", "score_percent": 82.0,
        "url": "https://x", "inconclusive": False,
        "skills_detail": {"matching": [{"job_skill": "Python"}], "missing": ["K8s"]},
    },
]
LETTER = "Dear Hiring Team,\nI am excited to apply.\nBest regards,\nJane Doe"


def test_split_long_message_no_split_when_short():
    assert _split_long_message("short body") == ["short body"]


def test_split_long_message_chunks_without_losing_content():
    para = "\n".join(f"paragraph {i} " + "x" * 100 for i in range(60))
    chunks = _split_long_message(para, limit=4000)
    assert len(chunks) > 1
    assert all(c for c in chunks), "no empty chunks"
    assert all(len(c) <= 4000 for c in chunks)
    # content preserved across chunks
    joined = "\n".join(chunks)
    assert "paragraph 0 " in joined and "paragraph 59 " in joined


def test_create_results_telegram_posts_to_bot_api():
    with patch("pipeline.send_results_telegram.requests.post") as mock_post:
        import pipeline.send_results_telegram as tg
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True, "message_id": 1}
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "123:TOKEN", "TELEGRAM_CHAT_ID": "42"},
            clear=False,
        ):
            result = create_results_telegram(FakeCv(), RANKED, LETTER)
    assert result["chat_id"] == "42"
    assert mock_post.call_count >= 1
    url = mock_post.call_args.args[0]
    payload = mock_post.call_args.kwargs["json"]
    assert url.startswith("https://api.telegram.org/bot123:TOKEN/sendMessage")
    assert payload["chat_id"] == "42"
    assert payload["text"]  # body rendered, not empty


def test_create_results_telegram_raises_without_env():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        try:
            create_results_telegram(FakeCv(), RANKED, LETTER)
        except FileNotFoundError as e:
            assert "TELEGRAM_BOT_TOKEN" in str(e)
        else:
            raise AssertionError("Expected FileNotFoundError without telegram env")


def test_ask_user_delivery_channel_normalizes_gmail():
    delivery_choice._last_delivery_channel = None
    with patch("builtins.input", return_value="mail"):
        out = ask_user_delivery_channel.func("")
    assert out == "Gmail"
    assert delivery_choice._last_delivery_channel == "gmail"


def test_ask_user_delivery_channel_normalizes_telegram():
    delivery_choice._last_delivery_channel = None
    with patch("builtins.input", return_value="tg"):
        out = ask_user_delivery_channel.func("")
    assert out == "Telegram"
    assert delivery_choice._last_delivery_channel == "telegram"


def test_ask_user_delivery_channel_retries_on_invalid():
    delivery_choice._last_delivery_channel = None
    with patch("builtins.input", side_effect=["carrier-pigeon", "gmail"]):
        out = ask_user_delivery_channel.func("")
    assert out == "Gmail"
    assert delivery_choice._last_delivery_channel == "gmail"


def test_send_results_telegram_errors_when_no_letter():
    telegram_module.job_evaluator._current_cv_info = FakeCv()
    telegram_module.cover_letter._last_cover_letter = None
    out = send_results_telegram.func("")
    assert out.startswith("Error: No cover letter")


def test_send_results_telegram_success_flow():
    """Happy path: letter present, env present, API returns 200 → success."""
    telegram_module.job_evaluator._current_cv_info = FakeCv()
    telegram_module.cover_letter._last_cover_letter = LETTER
    from agent.tools.job_evaluator import set_candidate_profile
    from agent.tools import job_evaluator
    job_evaluator._all_evaluations = [
        {"job_title": "AI Engineer", "company": "Acme", "score_percent": 82.0,
         "url": "https://x", "matching_skills": [], "missing_skills": [],
         "inconclusive": False},
    ]
    with patch("pipeline.send_results_telegram.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True, "message_id": 1}
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "123:B", "TELEGRAM_CHAT_ID": "7"},
            clear=False,
        ):
            out = send_results_telegram.func("")
    assert out.startswith("Telegram message sent successfully")


def test_aliases_cover_common_channels():
    assert _CHANNELS["email"] == "gmail"
    assert _CHANNELS["mail"] == "gmail"
    assert _CHANNELS["gmail"] == "gmail"
    assert _CHANNELS["tg"] == "telegram"
    assert _CHANNELS["telegram"] == "telegram"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All telegram/delivery tests passed.")