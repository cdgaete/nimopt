# Website scripts

| Script | What it writes |
|---|---|
| `llms.mjs` | `static/llms.txt` (the index) and `static/llms-full.txt` (every page under its route) |
| `skill.mjs` | `../../.claude/skills/nimopt/SKILL.md`, from the agents page |
| `readme.mjs` | `../../README.md`, from the front page |
| `watch.mjs` | nothing; it reruns the three generators on every change under `docs/` |

`pnpm generate` runs the three generators, `pnpm build` runs them and then
builds the site, and `pnpm watch` keeps every generated file current while
the docs are edited.
