from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .data import build_messages
from .utils import torch_dtype


def load_policy_and_ref(cfg: Config, device: str):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch_dtype(cfg.dtype)
    policy = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device)
    ref = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device)
    ref.eval()
    ref.requires_grad_(False)

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora)

    return tok, policy, ref


def encode_prompt(tok, question: str, cfg: Config, device: str) -> torch.Tensor:
    ids = tok.apply_chat_template(
        build_messages(question),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        truncation=True,
        max_length=cfg.max_prompt_len,
    )
    return ids.to(device)


@torch.no_grad()
def generate_group(policy, tok, prompt_ids: torch.Tensor, cfg: Config, greedy: bool = False):
    """Sample `group_size` completions for one prompt.

    Returns the full [G, L] token ids, an attention mask, and a completion mask
    that is 1 only on generated (non-prompt, non-pad) positions.
    """
    prompt_len = prompt_ids.shape[1]
    n = 1 if greedy else cfg.group_size
    out = policy.generate(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        do_sample=not greedy,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=n,
        pad_token_id=tok.pad_token_id,
    )
    attn = (out != tok.pad_token_id).long()
    # Generation appends to the right of a left-padded prompt, so completion
    # tokens are everything past prompt_len that is not padding.
    completion_mask = torch.zeros_like(out)
    completion_mask[:, prompt_len:] = 1
    completion_mask = completion_mask & (out != tok.pad_token_id)
    completion_text = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return out, attn, completion_mask, completion_text
