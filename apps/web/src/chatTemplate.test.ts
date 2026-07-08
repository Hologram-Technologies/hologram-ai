// The parametric chat-template path: any instruct model's OWN Jinja
// chat_template renders correctly, and a non-Jinja (`{prompt}`) or absent
// template falls back (returns null) — so the deterministic fixture is
// unaffected while arbitrary models get their real format + stop.
import { describe, expect, it } from "vitest";
import { renderChatTemplate } from "./chatTemplate";

// A ChatML template (Qwen/many instruct models), verbatim Jinja.
const CHATML =
  "{% for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

describe("renderChatTemplate", () => {
  it("renders a model's own ChatML template with the conversation + generation prompt", () => {
    const out = renderChatTemplate(
      CHATML,
      [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi" },
      ],
      { eosToken: "<|im_end|>" },
    );
    expect(out).not.toBeNull();
    expect(out!.prompt).toBe(
      "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
        "<|im_start|>user\nHi<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
    // The stop is the model's OWN eos, derived — never hard-coded per model.
    expect(out!.stop).toEqual(["<|im_end|>"]);
  });

  it("carries multi-turn history in order", () => {
    const out = renderChatTemplate(
      CHATML,
      [
        { role: "user", content: "1" },
        { role: "assistant", content: "one" },
        { role: "user", content: "2" },
      ],
      { eosToken: "<|im_end|>" },
    );
    expect(out!.prompt).toContain("<|im_start|>user\n1<|im_end|>");
    expect(out!.prompt).toContain("<|im_start|>assistant\none<|im_end|>");
    expect(out!.prompt.endsWith("<|im_start|>assistant\n")).toBe(true);
  });

  it("falls back (null) for a plain {prompt} template — the deterministic fixture", () => {
    expect(
      renderChatTemplate("User:\n{prompt}\nAssistant:\n", [{ role: "user", content: "hi" }]),
    ).toBeNull();
  });

  it("falls back (null) for an absent template (a base model)", () => {
    expect(renderChatTemplate(undefined, [{ role: "user", content: "hi" }])).toBeNull();
  });

  it("falls back (null) rather than throw on a malformed template", () => {
    expect(
      renderChatTemplate("{% for x in %}{{ x }}", [{ role: "user", content: "hi" }]),
    ).toBeNull();
  });
});
