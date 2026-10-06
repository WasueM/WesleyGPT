# Wesley wrote this
"""Chats -> token ids and labels for Qwen fine-tuning.

Only the final assistant answer is graded. Earlier turns are context, rendered
exactly as Qwen's template renders history at inference time, and the prompt
ends with the template's own generation prompt (which, for Qwen3.5, includes the
empty <think></think> block of non-thinking mode), so training and serving see
the same bytes before the answer.
"""
import torch

END_OF_TURN = "<|im_end|>"
IGNORE = -100  # label value the loss skips


def encode_example(tokenizer, messages):
    """{'input_ids', 'labels'} for one chat, grading only its last assistant message."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"a training chat must end with an assistant message, got {[m['role'] for m in messages]}")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(messages[-1]["content"] + END_OF_TURN, add_special_tokens=False)
    return {"input_ids": prompt_ids + answer_ids, "labels": [IGNORE] * len(prompt_ids) + answer_ids}


def collate(examples, pad_id):
    """Right-pad a list of encoded examples into tensors."""
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
