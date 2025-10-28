import os
import sys
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel

from state import initial_state, AssistantState
from graph import build_graph

console = Console()


def _provider_summary() -> str:
    base = os.environ.get("OPENAI_BASE_URL", "(default)")
    model = os.environ.get("MODEL", "(auto)")
    have = any([
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("OPENROUTER_API_KEY"),
        os.environ.get("GROQ_API_KEY"),
    ])
    return f"base={base}, model={model}, key={'yes' if have else 'no'}"


def main() -> None:
    console.print(Panel.fit("LangGraph Python Code Assistant (CLI)", title="Assistant"))
    console.print(f"Provider: {_provider_summary()}")

    graph = build_graph().compile()
    state: AssistantState = initial_state()

    console.print("Type your message. Commands: /reset, /examples, /config. Ctrl+C to exit.")

    try:
        while True:
            user = console.input("[bold cyan]> [/]")
            if not user.strip():
                continue
            if user.strip().lower() == "/reset":
                state = initial_state()
                console.print("[green]Conversation reset.[/]")
                continue
            if user.strip().lower() == "/config":
                console.print(_provider_summary())
                continue
            if user.strip().lower() == "/examples":
                _print_examples()
                continue

            state.setdefault("messages", []).append({"role": "user", "content": user})
            state = graph.invoke(state)
            last = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)
            if last:
                console.print(Panel.fit(last["content"], title="Assistant"))
    except KeyboardInterrupt:
        console.print("\n[bold]Bye![/]")


def _print_examples() -> None:
    path = os.path.join("examples", "seed_examples.jsonl")
    if not os.path.exists(path):
        console.print("No examples found.")
        return
    titles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = __import__("json").loads(line)
                titles.append(obj.get("title", "(untitled)"))
            except Exception:
                pass
    if not titles:
        console.print("No examples found.")
        return
    console.print("Examples:")
    for i, t in enumerate(titles, 1):
        console.print(f"  {i}. {t}")


if __name__ == "__main__":
    main()
