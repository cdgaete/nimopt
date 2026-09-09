import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "nimopt",
  tagline: "An LP/MILP builder in which a variable is a dimension",
  favicon: "img/favicon.ico",
  url: "https://example.invalid",
  baseUrl: "/",
  onBrokenLinks: "throw",
  markdown: {
    format: "detect",
    hooks: { onBrokenMarkdownLinks: "throw" },
  },
  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/",
          sidebarPath: "./sidebars.ts",
        },
        blog: false,
        theme: { customCss: "./src/css/custom.css" },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    navbar: {
      title: "nimopt",
      items: [
        { type: "docSidebar", sidebarId: "docs", position: "left", label: "Docs" },
        { to: "/playground", label: "Playground", position: "left" },
        { to: "/for-agents", label: "For agents", position: "right" },
      ],
    },
    footer: { style: "dark", links: [], copyright: "nimopt" },
    prism: { additionalLanguages: ["python"] },
  } satisfies Preset.ThemeConfig,
};

export default config;
