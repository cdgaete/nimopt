import { readFile, writeFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PAGE = join(here, "..", "docs", "index.md");
const OUT = join(here, "..", "..", "README.md");

// the page's routes are the site's; a reader on the repository has files
const ROUTES = {
  "/get-started": "website/docs/get-started/index.md",
  "/vocabulary": "website/docs/vocabulary/index.md",
  "/nimblend": "website/docs/nimblend/index.md",
  "/tutorial/sets-and-parameters": "website/docs/tutorial/sets-and-parameters.md",
  "/explanation/what-the-numbers-measure":
    "website/docs/explanation/what-the-numbers-measure.md",
  "/for-agents": "website/docs/for-agents.md",
};

// a route the repository cannot answer: the playground is the site running
// Python in the reader's browser, and a file is no substitute for it
const SITE_ONLY = new Set(["/playground"]);

function below(text) {
  if (!text.startsWith("---\n")) return text.trim();
  const parts = text.split("\n---\n");
  return parts.slice(1).join("\n---\n").trim();
}

function repoLinks(body) {
  // a site-only route keeps its words and loses its link, rather than
  // pointing a reader on the repository at a page that cannot work there
  return body
    .replace(/\[([^\]]*)\]\((\/[^)]*)\)/g, (whole, label, route) =>
      SITE_ONLY.has(route) ? label : whole,
    )
    .replace(/\]\((\/[^)]*)\)/g, (whole, route) => {
      const file = ROUTES[route];
      return file ? `](${file})` : whole;
    });
}

/** The README the page states, or the reason it cannot be stated. */
export function readme(page) {
  const body = repoLinks(below(page));
  const unresolved = [...body.matchAll(/\]\((\/[^)]*)\)/g)].map((m) => m[1]);
  if (unresolved.length) {
    throw new Error(`no repository path for ${unresolved.join(", ")}`);
  }
  return `${body}\n`;
}

const page = await readFile(PAGE, "utf8");
const text = readme(page);
await writeFile(OUT, text);
console.log(`README.md: ${text.length} characters`);
