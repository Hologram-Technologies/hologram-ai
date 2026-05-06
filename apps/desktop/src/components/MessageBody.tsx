import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface Props {
  text: string;
}

// Render assistant output as markdown with GFM (tables, fenced code) and math
// (`$…$`, `$$…$$`). User messages render the same way so pasted snippets,
// equations, and tables look right on the way in too.
export function MessageBody({ text }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
