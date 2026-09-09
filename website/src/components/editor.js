import React from "react";
import Editor from "react-simple-code-editor";
import { Highlight } from "prism-react-renderer";
import { usePrismTheme } from "@docusaurus/theme-common";
import styles from "./editor.module.css";

/** A Python editor coloured by the theme the site already renders code with. */
export default function Source({ value, onChange, disabled }) {
  const theme = usePrismTheme();
  return (
    <div className={styles.frame} style={{ background: theme.plain.backgroundColor }}>
      <Editor
        value={value}
        onValueChange={onChange}
        disabled={disabled}
        padding={16}
        textareaId="playground-source"
        textareaClassName={styles.area}
        preClassName={styles.pre}
        className={styles.editor}
        style={{ color: theme.plain.color }}
        highlight={(code) => (
          <Highlight code={code} language="python" theme={theme}>
            {({ tokens, getLineProps, getTokenProps }) =>
              tokens.map((line, at) => (
                <span {...getLineProps({ line, key: at })} key={at}>
                  {line.map((token, index) => (
                    <span {...getTokenProps({ token, key: index })} key={index} />
                  ))}
                  {"\n"}
                </span>
              ))
            }
          </Highlight>
        )}
      />
    </div>
  );
}
