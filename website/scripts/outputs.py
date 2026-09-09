"""What each documentation example prints, written below the fence that runs it.

A page states its examples; this states what they show. The region below a
fence is generated -- stripped and rewritten whole on every run -- so a page
is a function of its prose and its code, and running twice changes nothing.
"""

import contextlib
import io
import sys

from docs_blocks import FENCE, blocks, pages, strip_regions


def run(block):
    """What `block` prints, and the error it ends on."""
    buffer = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(block.body, "<docs>", "exec"), {"__name__": "__main__"})  # noqa: S102
    except Exception as raised:
        error = f"{type(raised).__name__}: {raised}"
    return buffer.getvalue(), error


def shown(block):
    """The summary and the text a block's region carries, or None for no region.

    A block that behaves against the form its fence states is refused rather
    than written down wrongly: the page would then state one thing and show
    another.
    """
    if block.form == "skip":
        return None
    printed, error = run(block)
    if block.form == "run":
        if error is not None:
            raise ValueError(f"line {block.line}: a plain block raised {error}")
        return ("Output", printed.strip()) if printed.strip() else None
    if error is None:
        raise ValueError(
            f"line {block.line}: a block stating raises={block.detail} did not"
        )
    if not error.startswith(f"{block.detail}:"):
        raise ValueError(
            f"line {block.line}: states raises={block.detail}, raised {error}"
        )
    body = f"{printed.strip()}\n{error}" if printed.strip() else error
    return f"Raises {block.detail}", body


def region(summary, text):
    """One generated region, delimited so the generator owns only what it wrote."""
    return (
        f"\n\n<!-- output -->\n<details open>\n<summary>{summary}</summary>\n\n"
        f"```text\n{text}\n```\n\n</details>\n<!-- /output -->"
    )


def rendered(text):
    """`text` with every region stripped and written again from a fresh run."""
    stripped = strip_regions(text)
    fences = list(FENCE.finditer(stripped))
    found = blocks(stripped)
    if len(fences) != len(found):
        raise ValueError("a fence lost its block")
    out, at = [], 0
    for fence, block in zip(fences, found):
        shows = shown(block)
        out.append(stripped[at : fence.end()])
        if shows is not None:
            out.append(region(*shows))
        at = fence.end()
    out.append(stripped[at:])
    return "".join(out)


def main():
    """Rewrite every page, reporting the ones that changed."""
    changed = 0
    for page in pages():
        text = page.read_text()
        fresh = rendered(text)
        if fresh != text:
            page.write_text(fresh)
            changed += 1
    total = sum(
        1 for page in pages() for b in blocks(page.read_text()) if b.output is not None
    )
    print(f"outputs: {total} regions, {len(pages())} pages, {changed} rewritten")


if __name__ == "__main__":
    sys.exit(main())
