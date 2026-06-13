import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import rehypePrettyCode, { type Options } from "rehype-pretty-code";
import { IdeaTimeChart } from "./idea-time-chart";

// Components usable from within MDX content.
const mdxComponents = { IdeaTimeChart };

// Dual-theme syntax highlighting — Shiki emits per-token --shiki-light /
// --shiki-dark CSS variables; globals.css switches between them by theme.
const prettyCodeOptions: Options = {
  theme: { light: "github-light", dark: "github-dark" },
  keepBackground: false,
};

export function Mdx({ source }: { source: string }) {
  return (
    <div className="prose">
      <MDXRemote
        source={source}
        components={mdxComponents}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkGfm],
            rehypePlugins: [rehypeSlug, [rehypePrettyCode, prettyCodeOptions]],
          },
        }}
      />
    </div>
  );
}
