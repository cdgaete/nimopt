# Website scripts

| Script | What it writes |
|---|---|
| `llms.mjs` | `static/llms.txt` (the index) and `static/llms-full.txt` (every page under its route) |
| `skill.mjs` | `../../.claude/skills/nimopt/SKILL.md`, from the agents page |
| `readme.mjs` | `../../README.md`, from the front page |
| `watch.mjs` | nothing; it reruns the three generators on every change under `docs/` |
| `serve.sh` | nothing; it launches llama.cpp for the chatbot |
| `chatbot.py` | nothing; it serves the manual as one context (below) |

`pnpm generate` runs the three generators, `pnpm build` runs them and then
builds the site, and `pnpm watch` keeps every generated file current while
the docs are edited.

## The documentation chatbot

The manual needs no retrieval: `static/llms-full.txt` is the whole
documentation, and every request carries it inside a fixed system prompt.
llama.cpp keeps the processed prefix cached in its slot, so the corpus is
prefilled once and each question pays only for its own tokens.

```sh
cd website
pnpm generate                     # a documentation change reaches the chatbot here
scripts/serve.sh                  # llama-server, one slot, cache under scripts/cache/
scripts/chatbot.py warm           # prefill once, save the slot to disk
scripts/chatbot.py ask "how do I read one row back out of a solved model?"
scripts/chatbot.py needle         # score the model against the corpus
```

The prompt is byte-identical on every request — the fixed rules, then the
corpus file — which is what makes the prefix cache hit. Serving a fresh
corpus therefore needs `pnpm generate` (or a running `pnpm watch`) followed
by `warm`; a server restart needs `warm --restore`, which loads the saved
slot instead of re-reading the manual.

`scripts/chatbot.py prompt` prints the prompt with a chars/4 token
estimate. Tokenizers run about a quarter above that estimate on this
corpus: the Qwen3.5 tokenizer measures the current manual at 65,783 tokens
against a 54,679 estimate, and `CTX` (default 73728) covers that prompt
plus roughly 8K of conversation. Grow `CTX` with the manual, and keep
answers short: a request larger than `CTX` is refused, not truncated.
`scripts/chatbot.py ask` with no question reads a conversation from stdin,
keeping turns in one slot. Expose the server beyond the machine with
`HOST=0.0.0.0 scripts/serve.sh`.

### Install

`llama-server` is a release download, not a package build:

```sh
ver=b10794
curl -sL -o /tmp/llama.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/$ver/llama-$ver-bin-ubuntu-x64.tar.gz"
mkdir -p ~/.local/share/llama.cpp/$ver ~/.local/bin
tar xzf /tmp/llama.tar.gz -C ~/.local/share/llama.cpp/$ver --strip-components 1
ln -sf ~/.local/share/llama.cpp/$ver/llama-server ~/.local/bin/llama-server
```

The tarball carries one CPU backend per microarchitecture and its shared
libraries next to the binary, so the symlink runs from any directory. The
model is a Hugging Face download:

```sh
mkdir -p ~/models && cd ~/models
curl -sL -o Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
MODEL=~/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf scripts/serve.sh
```

A cold prefill of the manual is CPU-bound and slows as the context grows:
measured at two hours for 65,787 tokens on an 8-core Ryzen 7 5800X —
36 tokens per second at the start, 9 at the end — with `--flash-attn on`,
16 threads and 2048-token batches. Every request after that reprocesses
only its own tokens (a cached question pays about ten seconds), but the
decode is the tax: about 5 seconds per token at full context, so a
64-token answer takes five minutes and questions should ask for little.
`THREADS` (default 16) should match the machine's logical cores.

Run `warm` detached (`setsid nohup ... &`) and leave it alone: a client
that dies mid-prefill wedges the slot, and the server then needs a
restart. The saved slot is 1.2 GB, restores in about fifteen seconds, and
is bound to the flags it was saved under — a slot saved without flash
attention restores into a flash-attention server only to be recomputed, so
save slots under the flags `serve.sh` runs.

### Model

**Qwen3.5-4B** (GGUF, Q4_K_M): three of every four layers run Gated
DeltaNet linear attention with a fixed-size state, so the cached
full-manual prefix costs far less RAM than a plain transformer of the same
size; Apache-2.0, context long past the manual, thinking mode available.
`--swa-full` stays in `serve.sh` for models with sliding-window attention;
this one has none, and llama.cpp disables the flag with a warning.
Alternates, in order:

| Model | Reason to prefer it | Cost |
|---|---|---|
| Jan-Code-4B | a coding fine-tune of the same family | check its attention type before trusting the cheap cache |
| Ministral 3 3B | the strongest math at this size | plain KV cache; verify the RAM cost at full-manual context |
| Phi-4-mini | best-in-class grade-school math | plain KV cache; fits a retrieval setup, not full-context |

### Validation

`scripts/chatbot.py needle` asks, in a fresh conversation each time, which
page documents a name that appears on exactly one page, at positions
spread across the whole corpus. An advertised context is not a used one:
the command exits nonzero under `--min-accuracy` (default 0.8), so a model
that reads only the early pages does not ship. Measured on Qwen3.5-4B
Q4_K_M over this manual: three of three pages located —
`/reference/explanation` from the early pages, `/tutorial/expressions`
from the middle, `/models/transport` from the end. Run it on the chosen
model and after any large documentation change.

GitHub Actions runs the same install and the same validation, on demand,
in `.github/workflows/chatbot.yml`.
