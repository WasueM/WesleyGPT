<!-- Wesley wrote this -->
# Experiments to try

Ideas Wesley wants to run on WesleyGPT, each with enough context to start
cold. Move an entry to a model card or `wesley/README.md` once it has run,
with its result, whichever way it came out.

## `<think>` and `</think>` as real special tokens

**Today:** the tags are plain text, 3 and 4 ordinary tokens each (`<` · `think` ·
`>\n`), so the model reuses embeddings it already learned in pretraining
(`wesleygpt/think.py`). This is the cheap choice, not the one the big labs make:
Qwen3 and DeepSeek-R1 give each tag its own token id.

**Experiment:** add `<|think_start|>` / `<|think_end|>` to the vocabulary and
learn their rows from scratch during SFT.

- Grow `wte` and `lm_head` by two rows (vocab 32,768 → 32,770; check whether
  nanochat pads the vocab to a multiple, which may leave spare rows to use
  instead), and add the names to `SPECIAL_TOKENS` in `nanochat/tokenizer.py`.
- Initialise the new rows from the mean of the existing embeddings rather than
  noise, and consider a higher learning rate for just those rows.
- Re-render the think SFT mix with the new tokens, retrain from base d12@2520
  with the same settings as d12-think@1038, and update `wesleygpt/reasoning.py`
  to split on token ids instead of text.
- Release format: the new tokenizer must ship with the model, and
  `tokenizer_config.json` must list the new specials.

**Compare against d12-think:** GSM8K (14.6%), ChatCORE (0.125), identity (45%),
how often it opens a think block on word problems, and the 3–4 tokens per tag
saved from the 2,048 window.

**Why it might not win:** two untrained rows have to be learned from ~270K
examples, while the text version starts from embeddings pretraining already
tuned. That trade-off is the point of the experiment.
