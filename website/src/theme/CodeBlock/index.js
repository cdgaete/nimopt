import React from "react";
import CodeBlock from "@theme-original/CodeBlock";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";
import styles from "./styles.module.css";

// the output regions below each fence are ```text, so they arrive as
// language-text and carry no button: only the source of an example does
const PYTHON = "language-python";

function source(children) {
  if (typeof children === "string") return children;
  if (Array.isArray(children) && children.every((part) => typeof part === "string")) {
    return children.join("");
  }
  return null;
}

export default function CodeBlockWithPlay(props) {
  const location = useLocation();
  const code = source(props.children);
  const runnable = (props.className ?? "").split(" ").includes(PYTHON) && code !== null;
  if (!runnable) {
    return <CodeBlock {...props} />;
  }
  const to = `/playground?code=${encodeURIComponent(code)}&from=${encodeURIComponent(location.pathname)}`;
  return (
    <div className={styles.block}>
      <CodeBlock {...props} />
      <div className={styles.bar}>
        <Link className={styles.play} to={to} title="Open this example in the playground">
          <span aria-hidden="true">▶</span> Run this example
        </Link>
      </div>
    </div>
  );
}
