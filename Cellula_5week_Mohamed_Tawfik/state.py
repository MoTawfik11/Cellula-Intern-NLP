from typing import Literal, TypedDict, List, Dict, Any

ChatState = Literal["generate_code", "explain_code", "fallback"]

class AssistantState(TypedDict, total=False):
    messages: List[Dict[str, Any]]  # list of {role, content}
    chat_state: ChatState
    examples: List[Dict[str, Any]]  # retrieved items
    params: Dict[str, Any]


def initial_state() -> AssistantState:
    return {
        "messages": [],
        "chat_state": "fallback",
        "examples": [],
        "params": {},
    }
