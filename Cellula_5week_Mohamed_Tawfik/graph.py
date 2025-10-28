from typing import Dict, Any, List

from langgraph.graph import StateGraph, START, END

from state import AssistantState
from routing import classify_intent
from retrieval import Retriever
from prompts import SYSTEM_PROMPT, format_generation_prompt, format_explanation_prompt
from llm import call_llm


retriever = Retriever()

def _append_message(state: AssistantState, role: str, content: str) -> AssistantState:
    msgs = list(state.get("messages", []))
    msgs.append({"role": role, "content": content})
    state["messages"] = msgs
    return state


def node_classify(state: AssistantState) -> AssistantState:
    last_user = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    state["chat_state"] = classify_intent(last_user)
    return state


def node_retrieve(state: AssistantState) -> AssistantState:
    last_user = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    results = retriever.search(last_user)
    state["examples"] = results
    return state


def node_call_llm(state: AssistantState) -> AssistantState:
    user_text = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    intent = state.get("chat_state", "fallback")
    examples = state.get("examples", [])

    if intent == "generate_code":
        msgs = format_generation_prompt(user_text, examples)
    elif intent == "explain_code":
        msgs = format_explanation_prompt(user_text, examples)
    else:
        msgs = [{"role": "user", "content": user_text}]

    assistant_reply = call_llm(msgs, system=SYSTEM_PROMPT)
    _append_message(state, "assistant", assistant_reply)
    return state


def build_graph() -> StateGraph:
    g = StateGraph(AssistantState)
    g.add_node("classify_intent", node_classify)

    def condition(state: AssistantState):
        if state.get("chat_state") in ("generate_code", "explain_code"):
            return "retrieve_examples"
        return "call_llm"

    g.add_node("retrieve_examples", node_retrieve)
    g.add_node("call_llm", node_call_llm)

    g.add_conditional_edges("classify_intent", condition, {
        "retrieve_examples": "retrieve_examples",
        "call_llm": "call_llm",
    })

    g.add_edge("retrieve_examples", "call_llm")
    g.add_edge(START, "classify_intent")
    g.add_edge("call_llm", END)
    return g
