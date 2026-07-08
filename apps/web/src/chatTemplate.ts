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
  } catch {
    return null;
  }
}
