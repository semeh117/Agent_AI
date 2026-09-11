"""
delivery_choice.py
--------------
Agent-facing tool: asks the candidate (a human at the CLI prompt) which
delivery channel they want for the ranked results + cover letter — a
Gmail draft or a Telegram message. The agent calls this between writing
the cover letter and delivering, then calls the matching transport tool
(send_results_draft for Gmail, send_results_telegram for Telegram).

The chosen channel is recorded in module state so the deterministic
completion (agent1._complete_deterministically) knows which
delivery to guarantee if the agent skips it — same pattern as
cover_letter_tool's _last_cover_letter.
"""

from langchain_core.tools import tool

_last_delivery_channel = None  # "gmail" | "telegram", once the user answers

# Normalized aliases so free-form human input still parses cleanly.
_CHANNELS = {
    "gmail": "gmail",
    "email": "gmail",
    "mail": "gmail",
    "telegram": "telegram",
    "tg": "telegram",
}


@tool
def ask_user_delivery_channel(_unused_input: str = "") -> str:
    """
    Asks the candidate WHICH way they want the ranked job matches and
    cover letter delivered: a Gmail draft or a Telegram message. Call this
    AFTER write_cover_letter, BEFORE send_results_draft /
    send_results_telegram. Prompts the human user at the console and waits
    for their answer.

    Input: not needed — pass an empty string.

    Output: "Gmail" or "Telegram" (the validated choice). Then call the
    matching deliver tool: send_results_draft for Gmail, or
    send_results_telegram for Telegram — exactly one of them.
    """
    global _last_delivery_channel

    while True:
        raw = input("How do you want the results + cover letter delivered? "
                    "(gmail / telegram): ").strip().lower()
        channel = _CHANNELS.get(raw)
        if channel:
            _last_delivery_channel = channel
            break
        print(f"  '{raw}' is not a valid choice. Type 'gmail' or 'telegram'.")

    return {"gmail": "Gmail", "telegram": "Telegram"}[channel]
