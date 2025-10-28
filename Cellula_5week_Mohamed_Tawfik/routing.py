from typing import Literal

ChatState = Literal["generate_code", "explain_code", "fallback"]

INTENT_RULES = {
    "generate_code": [
        "write", "generate", "create", "build", "implement", "code", "function",
    ],
    "explain_code": [
        "explain", "what does", "how works", "understand", "read", "comment", "describe",
    ],
}


def classify_intent(text: str) -> ChatState:
    if not text:
        return "fallback"
    lower = text.lower()
    for keyword in INTENT_RULES["explain_code"]:
        if keyword in lower:
            return "explain_code"
    for keyword in INTENT_RULES["generate_code"]:
        if keyword in lower:
            return "generate_code"
    return "fallback"
