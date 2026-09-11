import React, { useCallback, useEffect, useState } from "react";
import Layout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";
import BrowserOnly from "@docusaurus/BrowserOnly";
import Link from "@docusaurus/Link";
import Source from "../components/editor";
import { run, runtime } from "../components/pyodide";
import styles from "./playground.module.css";

const START = `import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))

m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

solution = m.solve()
print(solution.status, solution.objective)
print(solution.primal("x").to_dense())
`;

const KEEP = "nimopt.playground.source";

// a browser may refuse storage outright -- a private window, or site data
// turned off -- so every read and write states what it does when it cannot
function kept() {
  try {
    return window.localStorage.getItem(KEEP);
  } catch {
    return null;
  }
}

function keep(source) {
  try {
    window.localStorage.setItem(KEEP, source);
  } catch {
    // the editor still works for this visit; only the return to it is lost
  }
}

function asked() {
  return new URLSearchParams(window.location.search);
}

function seeded() {
  return asked().get("code") ?? kept() ?? START;
}

/** The page a play button came from, and the name to call it by. */
function origin() {
  const path = asked().get("from");
  if (!path || !path.startsWith("/")) return null;
  const last = path.replace(/\/$/, "").split("/").pop();
  const named = last ? last.replace(/-/g, " ") : "the documentation";
  return { path, name: named.charAt(0).toUpperCase() + named.slice(1) };
}

function Playground() {
  const wheels = [
    useBaseUrl("/wheels/nimblend-0.2.2-py3-none-any.whl"),
    useBaseUrl("/wheels/nimopt-0.2.2-py3-none-any.whl"),
  ];
  const [source, setSource] = useState(seeded);
  const [output, setOutput] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [came] = useState(origin);
  const [resumed, setResumed] = useState(() => {
    const held = kept();
    return held !== null && held !== START && !asked().has("code");
  });
  // an example opened from a play button is not the reader's work, and must not
  // replace the draft they left here; the first edit makes it theirs
  const [mine, setMine] = useState(() => !asked().has("code"));

  // on `source` rather than in the change handler, so Reset is written too
  useEffect(() => {
    if (mine) keep(source);
  }, [source, mine]);

  const onRun = useCallback(async () => {
    setBusy(true);
    setOutput("");
    try {
      const here = wheels.map((path) => new URL(path, window.location.href).href);
      const pyodide = await runtime(here, setNote);
      setNote("Running.");
      setOutput(await run(pyodide, source));
      setNote("");
    } catch (error) {
      setNote("");
      setOutput(String(error));
    } finally {
      setBusy(false);
    }
  }, [source]);

  return (
    <div className={styles.playground}>
      <p className={styles.lede}>
        Every example in the documentation runs here, and so does anything you write beside it. The code runs in this tab — nothing is uploaded, and nothing is downloaded until you press Run.
      </p>
      {came && (
        <p className={styles.resumed}>
          <Link to={came.path}>← Back to {came.name}</Link>
        </p>
      )}
      {resumed && (
        <p className={styles.resumed}>
          This is what you left here last time. Reset puts the example back.
        </p>
      )}
      <Source
        value={source}
        onChange={(next) => {
          setMine(true);
          setSource(next);
        }}
        disabled={busy}
      />
      <div className={styles.bar}>
        <button
          className="button button--primary"
          onClick={onRun}
          disabled={busy}
          type="button"
        >
          {busy ? "Working…" : "Run"}
        </button>
        <button
          className="button button--secondary"
          onClick={() => {
            setMine(true);
            setSource(START);
            setResumed(false);
            setOutput("");
          }}
          disabled={busy}
          type="button"
        >
          Reset
        </button>
        <span className={styles.note}>{note}</span>
      </div>
      {output !== "" && <pre className={styles.output}>{output}</pre>}
    </div>
  );
}

export default function Page() {
  return (
    <Layout title="Playground" description="Run and change the examples in your browser.">
      <main className="container margin-vert--lg">
        <h1>Playground</h1>
        <BrowserOnly fallback={<p>Loading…</p>}>{() => <Playground />}</BrowserOnly>
      </main>
    </Layout>
  );
}
