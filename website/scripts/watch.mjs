import { watch } from "node:fs";
import { spawnSync } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const docs = join(here, "..", "docs");
const generators = ["llms.mjs", "skill.mjs", "readme.mjs"];

function generate() {
  for (const script of generators) {
    spawnSync("node", [join(here, script)], { stdio: "inherit" });
  }
}

let waiting;
generate();
watch(docs, { recursive: true }, () => {
  clearTimeout(waiting);
  waiting = setTimeout(generate, 200);
});
console.log(`watching ${docs} -- every change rewrites llms.txt, the skill and the readme`);
