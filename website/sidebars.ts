import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docs: [
    "index",
    { type: "doc", id: "get-started/index", label: "Get started" },
    { type: "doc", id: "vocabulary/index", label: "Vocabulary" },
    {
      type: "category",
      label: "Tutorial",
      collapsed: false,
      items: [
        "tutorial/sets-and-parameters",
        "tutorial/variables",
        "tutorial/expressions",
        "tutorial/constraints",
        "tutorial/solving",
        "tutorial/reading-the-answer",
      ],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "guides/subsets",
        "guides/conditions",
        "guides/lags",
        "guides/fixed-members",
        "guides/bounds-from-parameters",
        "guides/coefficient-arithmetic",
        "guides/at-scale",
        "guides/highs-methods",
        "guides/saving-and-loading",
      ],
    },
    {
      type: "category",
      label: "Worked models",
      items: [
        "models/index",
        "models/dispatch",
        "models/transport",
        "models/sector",
        "models/storage",
        "models/nodal",
        "models/fleet",
        "models/commitment",
        "models/profiled",
        "models/expansion",
        "models/recourse",
      ],
    },
    {
      type: "category",
      label: "Reference",
      items: [
        "reference/model",
        "reference/sets",
        "reference/param",
        "reference/variable",
        "reference/expression",
        "reference/constraint",
        "reference/definition",
        "reference/explanation",
        "reference/inspection",
        "reference/solution",
        "reference/solvers",
        "reference/files",
      ],
    },
    {
      type: "category",
      label: "Explanation",
      items: [
        "explanation/a-variable-is-a-dimension",
        "explanation/expressions-are-symbolic",
        "explanation/the-array-is-the-matrix",
        "explanation/the-package-boundary",
        "explanation/what-the-numbers-measure",
      ],
    },
    {
      type: "category",
      label: "nimblend",
      items: ["nimblend/index", "nimblend/arrays", "nimblend/domains"],
    },
    "for-agents",
  ],
};

export default sidebars;
