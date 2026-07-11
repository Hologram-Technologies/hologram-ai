//! Prompt-lookup speculative-decode ACCEPTANCE measurement (ADR-0018 follow-on).
//!
//! Decides whether/how to default speculative decode on. It tokenizes realistic
//! chatbot (prompt + response) samples with the REAL tokenizer and simulates the
//! SHIPPED `prompt_lookup_draft` under greedy speculation — mean tokens advanced
//! per forward pass IS the speculative speedup (each pass ≈ one forward; the
//! verify m=K pass pools on substrate v0.8.2). For greedy this is EXACT when the
//! sequence is the model's own output; here it measures the drafter's predictive
//! power on the response text, which is what determines acceptance for the tasks
//! prompt-lookup targets (code edit, structured/format echo, quoting/RAG).
//!
//! Run: `cargo run -p hologram-ai --release --example spec_acceptance -- \
//!        [path/to/tokenizer.json]`
//! Default tokenizer: models/Qwen2.5-0.5B-Instruct/tokenizer.json.

use std::path::PathBuf;

use hologram_ai::speculative::prompt_lookup_draft;
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

/// Realistic (label, full prompt+response text). The response is what the model
/// streams; prompt-lookup drafts from the WHOLE realized sequence, so echoes of
/// the prompt (code edits, quoting) and self-repetition (formats, lists) recur.
const SAMPLES: &[(&str, &str)] = &[
    (
        "code-edit (echo+change)",
        "Refactor this function to add error handling:\n\
         fn parse_config(path: &str) -> Config {\n\
             let text = std::fs::read_to_string(path).unwrap();\n\
             toml::from_str(&text).unwrap()\n\
         }\n\n\
         Here is the refactored function with error handling:\n\
         fn parse_config(path: &str) -> Result<Config, ConfigError> {\n\
             let text = std::fs::read_to_string(path)?;\n\
             let config = toml::from_str(&text)?;\n\
             Ok(config)\n\
         }\n",
    ),
    (
        "structured/JSON (key echo)",
        "Convert this to JSON with fields name, role, active:\n\
         Alex is an admin and the account is enabled.\n\n\
         {\n\
           \"name\": \"Alex\",\n\
           \"role\": \"admin\",\n\
           \"active\": true\n\
         }\n",
    ),
    (
        "quote/RAG (verbatim span)",
        "Given the passage, answer: what is the ceiling?\n\
         Passage: The only ceiling is the wasm32 four gigabyte address space, a host \
         law, plus the substrate weight-tier pager for larger models.\n\n\
         Answer: According to the passage, the only ceiling is the wasm32 four \
         gigabyte address space, a host law, plus the substrate weight-tier pager \
         for larger models.\n",
    ),
    (
        "free-form prose (novel)",
        "Write a short reflection on the first snowfall.\n\n\
         The morning arrived hushed, as though the world had agreed to keep a \
         secret. Soft light pressed against the curtains and every familiar sound \
         had been folded into a deep, cottoned quiet that made even the kettle seem \
         to whisper its small complaint.\n",
    ),
];

const KS: &[usize] = &[2, 4, 8];

/// Greedy speculative simulation: mean tokens advanced per forward pass, and the
/// fraction of passes that drafted (proposed a non-empty continuation).
fn simulate(seq: &[i64], k: usize) -> (f64, f64) {
    let mut i = 1usize;
    let (mut passes, mut drafted) = (0usize, 0usize);
    while i < seq.len() {
        passes += 1;
        let draft = prompt_lookup_draft(&seq[..i], k);
        if draft.is_empty() {
            i += 1;
            continue;
        }
        drafted += 1;
        let mut acc = 0usize;
        while acc < draft.len() && i + acc < seq.len() && draft[acc] == seq[i + acc] {
            acc += 1;
        }
        i += acc + 1;
    }
    ((seq.len() - 1) as f64 / passes as f64, drafted as f64 / passes as f64)
}

fn main() {
    let tok_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Qwen2.5-0.5B-Instruct/tokenizer.json"));
    let tokenizer = match NativeTokenizer::from_tokenizer_json(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("could not load tokenizer {tok_path:?}: {e:#}");
            eprintln!("pass a tokenizer.json path as the first argument.");
            std::process::exit(1);
        }
    };

    println!("# Prompt-lookup speculative acceptance (tokenizer: {tok_path:?})");
    println!("# mean tokens/pass = speculative speedup in forward passes; drafted% = passes that proposed");
    println!("{:<26} {:>6} {:>7} {:>12} {:>9}", "sample", "K", "tokens", "tok/pass", "drafted%");

    let mut totals = vec![(0usize, 0usize, 0usize); KS.len()]; // (tokens, passes, drafted-passes)
    for (label, text) in SAMPLES {
        let ids: Vec<i64> = tokenizer.encode(text).into_iter().map(|t| t as i64).collect();
        for (ki, &k) in KS.iter().enumerate() {
            let (spd, drafted) = simulate(&ids, k);
            println!(
                "{:<26} {:>6} {:>7} {:>11.2}x {:>8.0}%",
                label, k, ids.len(), spd, drafted * 100.0
            );
            // Re-accumulate exact counts for a token-weighted overall mean.
            let mut i = 1usize;
            let (mut passes, mut dpasses) = (0usize, 0usize);
            while i < ids.len() {
                passes += 1;
                let d = prompt_lookup_draft(&ids[..i], k);
                if d.is_empty() {
                    i += 1;
                    continue;
                }
                dpasses += 1;
                let mut acc = 0usize;
                while acc < d.len() && i + acc < ids.len() && d[acc] == ids[i + acc] {
                    acc += 1;
                }
                i += acc + 1;
            }
            totals[ki].0 += ids.len().saturating_sub(1);
            totals[ki].1 += passes;
            totals[ki].2 += dpasses;
        }
        println!();
    }

    println!("# OVERALL (token-weighted across samples)");
    for (ki, &k) in KS.iter().enumerate() {
        let (tokens, passes, dpasses) = totals[ki];
        println!(
            "K={:<2}  {:.2}x tokens/pass   drafted on {:.0}% of passes",
            k,
            tokens as f64 / passes as f64,
            dpasses as f64 / passes.max(1) as f64 * 100.0
        );
    }
}
