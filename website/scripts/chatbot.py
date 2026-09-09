#!/usr/bin/env python3
"""The manual as one cached context: a fixed system prompt around the whole
llms-full.txt corpus, driven over HTTP against a llama.cpp server.

Every subcommand reads the corpus the website generates, so a documentation
change reaches the chatbot the moment `pnpm generate` (or a running
`pnpm watch`) rewrites it. The prompt is the fixed rules followed by the
corpus file, byte for byte on every request, which is what lets llama.cpp
reuse the processed prefix instead of re-reading fifty pages per question.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "static" / "llms-full.txt"
SLOT = "docs-slot.bin"

RULES = """You are a documentation assistant for nimopt. Your only source of
truth is the documentation provided below. Do not use outside knowledge about
nimopt or other packages unless the user explicitly asks for a comparison or
general programming help.

Rules:
1. Answer using only what is stated or directly inferable from the documentation below.
2. If the documentation does not cover the question, say so plainly. Do not guess or
fabricate function names, parameters, or behavior.
3. When the answer involves code, quote the exact function, class and parameter names as
they appear in the documentation. Do not invent syntax.
4. Keep answers concise. Every page below begins with its route as a heading, such as
# /reference/solvers. Cite the route an answer comes from so the user can verify it.
5. If a question is ambiguous, ask one clarifying question rather than guessing."""

BEGIN = "--- BEGIN DOCUMENTATION ---"
END = "--- END DOCUMENTATION ---"
ROUTE = re.compile(r"^# (/\S*)$", re.M)
TOKEN = re.compile(r"`([A-Za-z_][\w.]*)`")
THINKING = re.compile(r"<think>.*?</think>\s*", re.S)
QUESTION = (
    "Which documentation page documents `{}`? Reply with the page's route "
    "only, such as /reference/solvers."
)


def system_prompt():
    """The prompt every request carries: fixed rules, then the whole manual."""
    return f"{RULES}\n\n{BEGIN}\n{CORPUS.read_text()}\n{END}\n"


def estimate_tokens(text):
    """chars / 4: close enough to size a context window without a tokenizer."""
    return len(text) // 4


def post(url, payload, timeout=14400):
    """One JSON POST; the timeout covers a cold CPU prefill of the whole
    manual, which slows as the context grows and can run past an hour."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def reply(url, messages, max_tokens):
    """One chat completion, thinking off, its prefix cached."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = post(f"{url}/v1/chat/completions", payload)
    return THINKING.sub("", body["choices"][0]["message"]["content"]).strip()


def pages():
    """The corpus as (route, body) pairs, in the order the corpus carries."""
    parts = ROUTE.split(CORPUS.read_text())
    return list(zip(parts[1::2], parts[2::2]))


def needles():
    """(route, token) pairs where the token names exactly one page.

    A needle needs one verifiable answer, so a token only counts when the
    whole corpus mentions it on a single page: the model cannot score by
    paraphrasing a section it half remembers.
    """
    book = pages()
    found = []
    for route, body in book:
        if route == "/":
            continue
        for token in dict.fromkeys(TOKEN.findall(body)):
            if len(token) > 2 and sum(token in other for _, other in book) == 1:
                found.append((route, token))
                break
    return found


def wait_ready(url, deadline=120):
    """Block until the server answers health, so a client never races the
    model load."""
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError:
            pass
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f"{url} never became ready within {deadline}s")


def cmd_prompt(args):
    """Print the prompt (or write it out) with a token estimate on stderr."""
    prompt = system_prompt()
    if args.output:
        Path(args.output).write_text(prompt)
    else:
        sys.stdout.write(prompt)
    print(
        f"{len(prompt)} chars, about {estimate_tokens(prompt)} tokens (chars / 4)",
        file=sys.stderr,
    )


def cmd_warm(args):
    """Prefill the corpus once, then keep the slot on disk across restarts."""
    wait_ready(args.url)
    if args.restore:
        post(f"{args.url}/slots/0?action=restore", {"filename": SLOT})
    acknowledged = reply(
        args.url,
        [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": "Reply with the single word: ready."},
        ],
        8,
    )
    if args.no_save:
        print(f"{acknowledged or '(no reply)'} -- slot not saved")
        return
    post(f"{args.url}/slots/0?action=save", {"filename": SLOT})
    print(f"{acknowledged or '(no reply)'} -- slot saved as {SLOT}")


def cmd_ask(args):
    """Answer each question against the manual; no question reads stdin."""
    wait_ready(args.url)
    history = [{"role": "system", "content": system_prompt()}]
    questions = args.question if args.question else sys.stdin
    for question in questions:
        history.append({"role": "user", "content": question.strip()})
        answer = reply(args.url, history, args.max_tokens)
        history.append({"role": "assistant", "content": answer})
        print(answer, end="\n\n")


def cmd_needle(args):
    """Locate pages by a name only they carry, spread across the corpus.

    An advertised context is not a used one: this asks for one verifiable
    fact per position, a fresh conversation each time, and exits nonzero
    under --min-accuracy so a model that reads only the early pages does
    not ship.
    """
    wait_ready(args.url)
    found = needles()
    if args.count == 1:
        picks = [found[len(found) // 2]]
    else:
        step = (len(found) - 1) / (args.count - 1)
        picks = [found[round(at * step)] for at in range(args.count)]
    hits = 0
    for at, (route, token) in enumerate(picks):
        answer = reply(
            args.url,
            [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": QUESTION.format(token)},
            ],
            args.max_tokens,
        )
        located = route in answer
        hits += located
        mark = "ok" if located else "MISS"
        print(f"{mark} {at + 1}/{len(picks)} {route} {token} -> {answer}")
    accuracy = hits / len(picks)
    print(f"{hits}/{len(picks)} pages located, accuracy {accuracy:.2f}")
    sys.exit(0 if accuracy >= args.min_accuracy else 1)


def parser():
    """The command line: one subcommand per thing the harness does."""
    root = argparse.ArgumentParser(description="serve the nimopt manual as one context")
    where = argparse.ArgumentParser(add_help=False)
    where.add_argument("--url", default="http://127.0.0.1:8080")
    budget = argparse.ArgumentParser(add_help=False)
    budget.add_argument("--max-tokens", type=int, default=512)
    sub = root.add_subparsers(dest="command", required=True)

    prompt = sub.add_parser(
        "prompt", help="print the system prompt with a token estimate"
    )
    prompt.add_argument("-o", "--output")
    prompt.set_defaults(func=cmd_prompt)

    warm = sub.add_parser(
        "warm", parents=[where], help="prefill the corpus and save the slot"
    )
    warm.add_argument("--restore", action="store_true", help="load the slot first")
    warm.add_argument("--no-save", action="store_true", help="warm without saving")
    warm.set_defaults(func=cmd_warm)

    ask = sub.add_parser(
        "ask", parents=[where, budget], help="ask questions; none reads stdin"
    )
    ask.add_argument("question", nargs="*")
    ask.set_defaults(func=cmd_ask)

    needle = sub.add_parser(
        "needle", parents=[where, budget], help="score retrieval across the corpus"
    )
    needle.add_argument("--count", type=int, default=9)
    needle.add_argument("--min-accuracy", type=float, default=0.8)
    needle.set_defaults(func=cmd_needle)

    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
