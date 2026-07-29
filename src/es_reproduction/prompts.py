from __future__ import annotations

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def render_gsm8k_prompt(question: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question).strip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
