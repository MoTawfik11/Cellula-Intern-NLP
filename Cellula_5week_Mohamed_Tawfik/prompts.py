from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Be concise, correct, and pragmatic. "
    "When generating code, provide runnable snippets and short commentary."
)


def format_generation_prompt(user_text: str, examples: List[Dict]) -> List[Dict[str, str]]:
    examples_text = "\n\n".join(
        f"Example: {ex.get('title','')}\n{ex.get('description','')}\n```python\n{ex.get('code','')}\n```"
        for ex in examples
    )
    content = (
        f"User need (generate code):\n{user_text}\n\n"
        f"Relevant examples:\n{examples_text if examples_text else 'None'}\n\n"
        "Produce the best possible Python solution. Start with a brief plan, then code, then a short explanation."
    )
    return [{"role": "user", "content": content}]


def format_explanation_prompt(user_text: str, examples: List[Dict]) -> List[Dict[str, str]]:
    examples_text = "\n\n".join(
        f"Reference: {ex.get('title','')}\n{ex.get('description','')}\n"
        for ex in examples
    )
    content = (
        f"Explain the following Python code or concept clearly:\n{user_text}\n\n"
        f"Helpful references (if any):\n{examples_text if examples_text else 'None'}\n\n"
        "Explain line-by-line when appropriate, include complexity where relevant, and suggest improvements."
    )
    return [{"role": "user", "content": content}]
