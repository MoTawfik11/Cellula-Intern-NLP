# generator.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Public model, works without HF login
MODEL_NAME = "Salesforce/codegen-350M-mono"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CodeGenerator:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        if device == "cuda":
            self.model.to("cuda")
        else:
            self.model.to("cpu")
        # Use transformers pipeline for simpler generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device=="cuda" else -1
        )

    def generate(self, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95):
        output = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,  # only new tokens
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return output[0]["generated_text"]

if __name__ == "__main__":
    g = CodeGenerator()
    prompt = "### Task: implement function add(a,b):\n# return sum\n\ndef add(a,b):"
    s = g.generate(prompt)
    print(s)
