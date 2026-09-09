import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const DOCS = join(here, "..", "docs");
const STATIC = join(here, "..", "static");

const ORDER = [
  ".",
  "get-started",
  "vocabulary",
  "reference",
  "nimblend",
  "explanation",
  "tutorial",
  "guides",
  "models",
];
const TITLES = {
  "get-started": "Get started",
  vocabulary: "Vocabulary",
  tutorial: "Tutorial",
  guides: "Guides",
  models: "Worked models",
  reference: "Reference",
  nimblend: "nimblend",
  explanation: "Explanation",
  ".": "For agents",
};

async function markdown(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await markdown(path)));
    else if (entry.name.endsWith(".md")) out.push(path);
  }
  return out.sort();
}

function frontmatter(text) {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(text);
  if (!match) return { meta: {}, body: text };
  const meta = {};
  for (const line of match[1].split("\n")) {
    const at = line.indexOf(":");
    if (at > 0) meta[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  return { meta, body: text.slice(match[0].length) };
}

function route(page) {
  if (page.meta.slug) return page.meta.slug;
  const rel = relative(DOCS, page.path).replace(/\.md$/, "").split("\\").join("/");
  return "/" + rel.replace(/(^|\/)index$/, "");
}

function section(path) {
  const rel = relative(DOCS, path).split("\\").join("/");
  return rel.includes("/") ? rel.slice(0, rel.indexOf("/")) : ".";
}

function rank(page) {
  const at = ORDER.indexOf(section(page.path));
  return at === -1 ? ORDER.length : at;
}

function position(page) {
  const stated = Number(page.meta.sidebar_position);
  return Number.isFinite(stated) ? stated : Number.MAX_SAFE_INTEGER;
}

function byPath(a, b) {
  return a.path < b.path ? -1 : a.path > b.path ? 1 : 0;
}

const files = await markdown(DOCS);
const read = (
  await Promise.all(
    files.map(async (path) => ({ path, ...frontmatter(await readFile(path, "utf8")) })),
  )
).sort((a, b) => rank(a) - rank(b) || position(a) - position(b) || byPath(a, b));

const index = [
  "# nimopt",
  "",
  "> An LP/MILP builder in which a variable is a dimension. A constraint is",
  "> an array over its free sets crossed with the column space, so the array",
  "> is the matrix. nimblend, the labeled sparse array it is built on, is",
  "> documented under its own section.",
  "",
];
for (const name of ORDER) {
  const group = read.filter((page) => section(page.path) === name);
  if (!group.length) continue;
  index.push(`## ${TITLES[name]}`, "");
  for (const page of group) {
    const title = page.meta.title ?? route(page);
    const description = page.meta.description ?? "";
    index.push(`- [${title}](${route(page)}): ${description}`);
  }
  index.push("");
}

const REGION =
  /\n\n<!-- output -->\n<details open>\n<summary>([^\n<]*)<\/summary>\n\n(```text\n[\s\S]*?\n```)\n\n<\/details>\n<!-- \/output -->/g;

const MARKERS = /^<!-- \/?(?:surface|refusals|options) -->\n/gm;

function plain(body) {
  return body
    .replace(REGION, (_, summary, fence) => `\n\n${summary}:\n\n${fence}`)
    .replace(MARKERS, "");
}

const full = read.map(
  (page) => `# ${route(page)}\n\n${plain(page.body).replace(/<[A-Z][^>]*>/g, "").trim()}\n`,
);

await mkdir(STATIC, { recursive: true });
await writeFile(join(STATIC, "llms.txt"), index.join("\n"));
await writeFile(join(STATIC, "llms-full.txt"), full.join("\n---\n\n"));
console.log(`llms.txt and llms-full.txt: ${read.length} pages`);
