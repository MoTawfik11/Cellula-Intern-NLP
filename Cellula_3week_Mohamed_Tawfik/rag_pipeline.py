# rag_pipeline.py
from retriever import Retriever
from generator import CodeGenerator
import re

INSTRUCTION_TEMPLATE = """
You are an expert Python developer. Given the task below and example solutions, write a complete, fully working Python function. 
Only return the function code (no explanations, no tests).

Task:
{task_prompt}

Now write the function:
"""

def run_rag(task_prompt, top_k=3, max_new_tokens=256):
    retriever = Retriever()
    generator = CodeGenerator()

    _ = retriever.retrieve(task_prompt, top_k=top_k)  # optional retrieval

    prompt = INSTRUCTION_TEMPLATE.format(task_prompt=task_prompt)
    generated_text = generator.generate(prompt, max_new_tokens=max_new_tokens)

    # Robust regex: capture first function and all indented lines
    func_match = re.search(
        r"^(def\s+\w+\s*\(.*?\):\s*(?:\n(?: {4}|\t).+)+)", 
        generated_text, 
        re.MULTILINE
    )

    if func_match:
        return func_match.group(1) + "\n"

    # fallback
    return generated_text.strip()

if __name__ == "__main__":
    task = "Write a function `is_prime(n)` that returns True if n is prime, False otherwise."
    print(run_rag(task))
