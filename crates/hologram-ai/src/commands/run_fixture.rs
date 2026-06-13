//! `hologram-ai run-fixture` — execute an embedded fixture from a `.holo`.
//!
//! Loads the archive, reads the embedded deterministic fixture (manifest +
//! typed inputs + expected outputs), executes the model on those inputs, and
//! verifies the resulting output bytes and κ-labels match the embedded witness.
//! For logits-shaped outputs, it can also decode top-k token predictions using
//! the archive's embedded tokenizer.

use anyhow::{bail, Context, Result};
use clap::Args;
use hologram_ai_tokenizer::Tokenizer;
use std::cmp::Ordering;
use std::path::PathBuf;

use crate::fixture::{
    blake3_kappa, dtype_name, read_embedded_fixture, EmbeddedFixtureBundle, TensorManifest,
};
use crate::runner::HoloRunner;

#[derive(Args, Debug)]
pub struct RunFixtureArgs {
    /// Path to the `.holo` file carrying an embedded fixture.
    pub file: PathBuf,
    /// Print a short typed preview of each verified output.
    #[arg(long)]
    pub verbose: bool,
    /// Decode the top-k token predictions from a logits-shaped output using the
    /// archive's embedded tokenizer.
    #[arg(long, value_name = "K")]
    pub decode_top_k: Option<usize>,
    /// Restrict token decoding to specific flattened sequence positions.
    #[arg(long = "decode-position", value_name = "INDEX")]
    pub decode_positions: Vec<usize>,
    /// Output tensor index to decode as logits. Defaults to 0.
    #[arg(long, default_value_t = 0, value_name = "INDEX")]
    pub decode_output_index: usize,
    /// Decode only `[MASK]` positions from a logits-shaped output using the
    /// archive's embedded tokenizer.
    #[arg(long, value_name = "K")]
    pub masked_top_k: Option<usize>,
}

pub fn execute(args: RunFixtureArgs) -> Result<()> {
    let mut runner = HoloRunner::from_path(&args.file, None)
        .with_context(|| format!("loading model {:?}", args.file))?;
    let embedded = read_embedded_fixture(&runner)
        .with_context(|| format!("reading embedded fixture from {:?}", args.file))?;

    println!(
        "Loaded embedded fixture {:?}: preset {}, {} input(s), {} output(s)",
        args.file,
        embedded.manifest.preset,
        embedded.inputs.len(),
        embedded.outputs.len()
    );

    let input_refs: Vec<&[u8]> = embedded.inputs.iter().map(Vec::as_slice).collect();
    let outputs = runner
        .execute(&input_refs)
        .context("executing embedded fixture inputs")?;
    verify_output_count(outputs.len(), embedded.outputs.len())?;

    for (index, actual) in outputs.iter().enumerate() {
        let expected_manifest = &embedded.manifest.outputs[index];
        let expected_bytes = &embedded.outputs[index];
        let actual_kappa = blake3_kappa(&actual.bytes);

        if actual.bytes != *expected_bytes {
            bail!(
                "fixture output[{index}] bytes differ from the embedded witness ({} bytes vs {} bytes)",
                actual.bytes.len(),
                expected_bytes.len()
            );
        }
        if actual_kappa != expected_manifest.kappa {
            bail!(
                "fixture output[{index}] re-addressed to {actual_kappa}, expected {}",
                expected_manifest.kappa
            );
        }

        println!(
            "  output[{index}] verified: {} × {} ({} bytes) κ={}",
            dtype_name(expected_manifest.dtype_tag),
            expected_manifest.element_count,
            actual.bytes.len(),
            expected_manifest.kappa
        );
        if args.verbose {
            println!(
                "    {}",
                preview(&actual.bytes, expected_manifest.dtype_tag)
            );
        }
    }

    if let Some(top_k) = args.decode_top_k {
        decode_logits_output(
            &runner,
            &embedded,
            &outputs,
            top_k,
            &args.decode_positions,
            args.decode_output_index,
        )?;
    }
    if let Some(top_k) = args.masked_top_k {
        decode_masked_tokens(
            &runner,
            &embedded,
            &outputs,
            top_k,
            args.decode_output_index,
        )?;
    }

    Ok(())
}

fn verify_output_count(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    bail!("embedded fixture expects {expected} output(s), but execution produced {actual}")
}

fn preview(bytes: &[u8], dtype: u8) -> String {
    const MAX: usize = 16;
    match dtype {
        8 => fmt_vals(bytes, 4, MAX, |chunk| {
            f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")) as f64
        }),
        9 => fmt_vals(bytes, 8, MAX, |chunk| {
            f64::from_le_bytes(chunk.try_into().expect("8-byte f64 chunk"))
        }),
        4 => fmt_vals(bytes, 4, MAX, |chunk| {
            i32::from_le_bytes(chunk.try_into().expect("4-byte i32 chunk")) as f64
        }),
        5 => fmt_vals(bytes, 8, MAX, |chunk| {
            i64::from_le_bytes(chunk.try_into().expect("8-byte i64 chunk")) as f64
        }),
        _ => hex_preview(bytes, MAX),
    }
}

fn fmt_vals(bytes: &[u8], stride: usize, max: usize, decode: impl Fn(&[u8]) -> f64) -> String {
    let values: Vec<String> = bytes
        .chunks_exact(stride)
        .take(max)
        .map(|chunk| format!("{:.6}", decode(chunk)))
        .collect();
    let ellipsis = if bytes.len() / stride > max {
        ", …"
    } else {
        ""
    };
    format!("[{}{}]", values.join(", "), ellipsis)
}

fn hex_preview(bytes: &[u8], max: usize) -> String {
    let take = bytes.len().min(max);
    let preview = bytes[..take]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("");
    let ellipsis = if bytes.len() > take { "…" } else { "" };
    format!("0x{preview}{ellipsis}")
}

fn decode_logits_output(
    runner: &HoloRunner,
    embedded: &EmbeddedFixtureBundle,
    outputs: &[hologram_exec::OutputBuffer],
    top_k: usize,
    decode_positions: &[usize],
    decode_output_index: usize,
) -> Result<()> {
    let top_k = validate_top_k(top_k)?;
    let tokenizer = runner.embedded_tokenizer()?;
    let decoded = decoded_logits(embedded, outputs, decode_output_index)?;
    let positions = decode_row_positions(decoded.rows.len(), decode_positions)?;
    let input_tokens = find_input_token_ids(embedded)
        .transpose()?
        .unwrap_or_default();

    println!(
        "Decoded output[{decode_output_index}] as logits: {} row(s), vocab {}",
        decoded.rows.len(),
        decoded.vocab
    );
    for &position in &positions {
        let row = decoded.rows[position];
        let input_label = input_token_label(&tokenizer, input_tokens.get(position).copied());
        println!("  pos[{position}] input={input_label}");
        for candidate in top_k_predictions(row, top_k) {
            let rendered = render_token(&tokenizer, candidate.token_id);
            println!(
                "    {:>5} {:>10.6}  {}",
                candidate.token_id, candidate.logit, rendered
            );
        }
    }

    Ok(())
}

fn decode_masked_tokens(
    runner: &HoloRunner,
    embedded: &EmbeddedFixtureBundle,
    outputs: &[hologram_exec::OutputBuffer],
    top_k: usize,
    decode_output_index: usize,
) -> Result<()> {
    let top_k = validate_top_k(top_k)?;
    let tokenizer = runner.embedded_tokenizer()?;
    let decoded = decoded_logits(embedded, outputs, decode_output_index)?;
    let input_tokens = find_input_token_ids(embedded)
        .transpose()?
        .context("masked-token view requires an input_ids tensor in the embedded fixture")?;
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .context("embedded tokenizer does not define a [MASK] token")?;
    let positions = masked_positions(&input_tokens, mask_token_id);

    if positions.is_empty() {
        bail!(
            "fixture input_ids contains no [MASK] token; export a masked fixture preset or use --decode-position"
        );
    }
    if positions
        .iter()
        .any(|&position| position >= decoded.rows.len())
    {
        bail!(
            "fixture input_ids has {} token position(s), but decoded output only has {} row(s)",
            input_tokens.len(),
            decoded.rows.len()
        );
    }

    println!(
        "Masked-token view output[{decode_output_index}]: {} mask position(s), vocab {}",
        positions.len(),
        decoded.vocab
    );
    for position in positions {
        println!("  mask[{position}]");
        for candidate in top_k_predictions(decoded.rows[position], top_k) {
            let rendered = render_token(&tokenizer, candidate.token_id);
            println!(
                "    {:>5} {:>10.6}  {}",
                candidate.token_id, candidate.logit, rendered
            );
        }
    }

    Ok(())
}

fn validate_top_k(top_k: usize) -> Result<usize> {
    if top_k == 0 {
        bail!("--decode-top-k must be at least 1");
    }
    Ok(top_k)
}

fn logits_rows<'a>(bytes: &'a [u8], tensor: &TensorManifest) -> Result<Vec<&'a [f32]>> {
    if tensor.dtype_tag != 8 {
        bail!(
            "output {:?} is not f32 logits (dtype tag {}); cannot decode tokens",
            tensor.name,
            tensor.dtype_tag
        );
    }
    let vocab = logits_vocab_size(tensor)?;
    let values: &[f32] = bytemuck::try_cast_slice(bytes)
        .map_err(|_| anyhow::anyhow!("output {:?} is not a valid f32 buffer", tensor.name))?;
    if !values.len().is_multiple_of(vocab) {
        bail!(
            "output {:?} has {} f32 values, not divisible by vocab size {vocab}",
            tensor.name,
            values.len()
        );
    }
    Ok(values.chunks_exact(vocab).collect())
}

fn logits_vocab_size(tensor: &TensorManifest) -> Result<usize> {
    tensor
        .shape
        .last()
        .copied()
        .filter(|&vocab| vocab > 0)
        .context("logits output must have a non-empty last dimension")
}

struct DecodedLogits<'a> {
    rows: Vec<&'a [f32]>,
    vocab: usize,
}

fn decoded_logits<'a>(
    embedded: &'a EmbeddedFixtureBundle,
    outputs: &'a [hologram_exec::OutputBuffer],
    decode_output_index: usize,
) -> Result<DecodedLogits<'a>> {
    let tensor = embedded
        .manifest
        .outputs
        .get(decode_output_index)
        .with_context(|| {
            format!(
                "decode output index {decode_output_index} out of range (fixture has {} output(s))",
                embedded.manifest.outputs.len()
            )
        })?;
    let output = outputs
        .get(decode_output_index)
        .with_context(|| format!("execution did not produce output[{decode_output_index}]"))?;
    let rows = logits_rows(output.bytes.as_slice(), tensor)?;
    let vocab = logits_vocab_size(tensor)?;
    Ok(DecodedLogits { rows, vocab })
}

fn decode_row_positions(row_count: usize, requested: &[usize]) -> Result<Vec<usize>> {
    if requested.is_empty() {
        return Ok((0..row_count).collect());
    }
    let mut positions = Vec::with_capacity(requested.len());
    for &position in requested {
        if position >= row_count {
            bail!("decode position {position} out of range for {row_count} row(s)");
        }
        positions.push(position);
    }
    Ok(positions)
}

fn find_input_token_ids(embedded: &EmbeddedFixtureBundle) -> Option<Result<Vec<u32>>> {
    embedded
        .manifest
        .inputs
        .iter()
        .position(|tensor| tensor.name == "input_ids")
        .map(|index| {
            decode_integer_tokens(&embedded.inputs[index], &embedded.manifest.inputs[index])
        })
}

fn masked_positions(tokens: &[u32], mask_token_id: u32) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == mask_token_id).then_some(index))
        .collect()
}

fn decode_integer_tokens(bytes: &[u8], tensor: &TensorManifest) -> Result<Vec<u32>> {
    match tensor.dtype_tag {
        1 => Ok(bytes.iter().map(|&value| value as u32).collect()),
        2 => Ok(bytes
            .iter()
            .map(|&value| (value as i8) as i32 as u32)
            .collect()),
        4 => bytes
            .chunks_exact(4)
            .map(|chunk| {
                let value = i32::from_le_bytes(chunk.try_into().expect("4-byte i32 chunk"));
                u32::try_from(value)
                    .with_context(|| format!("input_ids contains negative INT32 token id {value}"))
            })
            .collect(),
        5 => bytes
            .chunks_exact(8)
            .map(|chunk| {
                let value = i64::from_le_bytes(chunk.try_into().expect("8-byte i64 chunk"));
                u32::try_from(value)
                    .with_context(|| format!("input_ids contains negative INT64 token id {value}"))
            })
            .collect(),
        other => bail!("input_ids has unsupported integer dtype tag {other}"),
    }
}

fn input_token_label(tokenizer: &impl Tokenizer, token_id: Option<u32>) -> String {
    match token_id {
        Some(token_id) => format!("{token_id} ({})", render_token(tokenizer, token_id)),
        None => "<unknown>".to_string(),
    }
}

fn render_token(tokenizer: &impl Tokenizer, token_id: u32) -> String {
    let decoded = tokenizer.decode(&[token_id]);
    let raw = tokenizer.id_to_token(token_id).unwrap_or("<unknown>");
    if decoded == raw {
        format!("{decoded:?}")
    } else {
        format!("{decoded:?} raw={raw:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TokenCandidate {
    token_id: u32,
    logit: f32,
}

fn top_k_predictions(row: &[f32], top_k: usize) -> Vec<TokenCandidate> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        row[right]
            .partial_cmp(&row[left])
            .unwrap_or(Ordering::Equal)
    });
    indices
        .into_iter()
        .take(top_k.min(row.len()))
        .map(|index| TokenCandidate {
            token_id: index as u32,
            logit: row[index],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTokenizer;

    impl Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }

        fn decode(&self, tokens: &[u32]) -> String {
            tokens
                .iter()
                .map(|token| format!("tok{token}"))
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn vocab_size(&self) -> usize {
            8
        }

        fn id_to_token(&self, id: u32) -> Option<&str> {
            match id {
                0 => Some("[PAD]"),
                1 => Some("hello"),
                2 => Some("world"),
                3 => Some("mask"),
                _ => None,
            }
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            match _token {
                "[MASK]" => Some(3),
                _ => None,
            }
        }
    }

    #[test]
    fn top_k_predictions_sort_descending() {
        let row = [-1.0, 4.0, 3.0, 7.0];
        let top = top_k_predictions(&row, 3);
        assert_eq!(
            top,
            vec![
                TokenCandidate {
                    token_id: 3,
                    logit: 7.0,
                },
                TokenCandidate {
                    token_id: 1,
                    logit: 4.0,
                },
                TokenCandidate {
                    token_id: 2,
                    logit: 3.0,
                },
            ]
        );
    }

    #[test]
    fn decode_integer_tokens_reads_i64_inputs() {
        let tensor = TensorManifest {
            index: 0,
            name: "input_ids".into(),
            dtype_tag: 5,
            dtype_name: "i64".into(),
            element_count: 3,
            shape: vec![1, 3],
            bytes_file: "inputs/input_0_input_ids.bin".into(),
            kappa: "blake3:test".into(),
        };
        let bytes = [
            101i64.to_le_bytes(),
            2023i64.to_le_bytes(),
            102i64.to_le_bytes(),
        ]
        .concat();
        assert_eq!(
            decode_integer_tokens(&bytes, &tensor).expect("decode integer tokens"),
            vec![101, 2023, 102]
        );
    }

    #[test]
    fn render_token_shows_decoded_and_raw_forms() {
        let rendered = render_token(&MockTokenizer, 1);
        assert!(rendered.contains("tok1"));
        assert!(rendered.contains("hello"));
    }

    #[test]
    fn masked_positions_finds_all_mask_tokens() {
        assert_eq!(masked_positions(&[101, 3, 2003, 3, 102], 3), vec![1, 3]);
    }
}
