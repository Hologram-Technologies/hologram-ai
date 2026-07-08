// Rendering a conversation with a MODEL'S OWN chat template — the parametric
// path that lets ANY instruct model work without a hard-coded per-model
// template. A model's `tokenizer_config.json` ships a Jinja `chat_template`
// (the exact one `transformers` uses); we render it here with the conversation.

import { Template } from "@huggingface/jinja";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

/**
 * Render `messages` with the model's own Jinja `chat_template`, returning the
 * full prompt and the model's stop token. Returns `null` when the template is
 * not Jinja (`{{ … }}` / `{% … %}`) — a plain `{prompt}`-style template or none
 * — so the caller falls back to the catalogue's prompt-template plumbing (and
 * the deterministic `{prompt}` fixture is unaffected).
 */
export function renderChatTemplate(
  chatTemplate: string | undefined,
  messages: ChatMessage[],
  opts: { eosToken?: string; bosToken?: string } = {},
): { prompt: string; stop: string[] } | null {
  if (!chatTemplate || (!chatTemplate.includes("{{") && !chatTemplate.includes("{%"))) {
    return null;
  }
  try {
    const template = new Template(chatTemplate);
    const prompt = template.render({
      messages,
      add_generation_prompt: true,
      bos_token: opts.bosToken ?? "",
      eos_token: opts.eosToken ?? "",
    });
    // The model's own eos ends its assistant turn (the engine also stops on the
    // tokenizer's eos id — this is the text backstop).
    return { prompt, stop: opts.eosToken ? [opts.eosToken] : [] };
  } catch (e) {
    // The template DECLARED itself Jinja (it passed the `{{`/`{%` guard) but
    // failed to compile/render. Degrade to the caller's fallback so the chat
    // survives — but SURFACE it: a silent null here would let a real instruct
    // model masquerade as a base model with generic stops. The "not Jinja /
    // absent" case above returns null without a warning (a legitimate, expected
    // fallback); only an actual render failure is noisy.
    console.warn(`chat_template failed to render, falling back to plain prompt: ${e}`);
    return null;
  }
}
