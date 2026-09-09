import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PAGE = join(here, "..", "docs", "for-agents.md");
const OUT = join(here, "..", "..", ".claude", "skills", "nimopt", "SKILL.md");

function below(text) {
  if (!text.startsWith("---\n")) return text.trim();
  const parts = text.split("\n---\n");
  return parts.slice(1).join("\n---\n").trim();
}

const text = await readFile(PAGE, "utf8");
const body = below(text);

const head = [
  "---",
  "name: nimopt",
  "description: Use when writing, reading or debugging a model built with nimopt or nimblend -- declaring sets, parameters, variables, expressions and constraints, or reading a solution.",
  "---",
  "",
].join("\n");

await mkdir(dirname(OUT), { recursive: true });
await writeFile(OUT, `${head}${body}\n`);
console.log(`SKILL.md: ${body.length} characters`);
