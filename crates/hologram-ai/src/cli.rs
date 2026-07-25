//! Command handlers behind the `hologram ai …` CLI group.
//!
//! The `hologram` CLI (in the Hologram repo) parses argv and delegates to
//! these handlers; there is no `hologram-ai` binary. Handlers never print
//! directly: they return a [`CliOutput`] with an exit code and a text or
//! JSON payload. JSON is emitted by a minimal internal writer (no serde
//! dependency).

use std::fmt::Write as _;
use std::path::PathBuf;

use hologram_ai_core::{AiError, AiResult, FinishReason, InferenceRequest};
use hologram_ai_huggingface::{HuggingFaceProvider, ModelSourceProvider, SourceRequest};

use crate::{Application, Compiler, HuggingFaceSource, LocalSource, Source, DEFAULT_ENTRY};

/// The result of a command: process exit code plus payload to print.
#[derive(Debug)]
pub struct CliOutput {
    /// 0 on success; otherwise the stable `ErrorCategory` code.
    pub code: i32,
    pub text: String,
}

impl CliOutput {
    fn ok(text: String) -> Self {
        Self { code: 0, text }
    }
}

impl From<AiError> for CliOutput {
    fn from(e: AiError) -> Self {
        Self {
            code: i32::from(e.code()),
            text: e.to_string(),
        }
    }
}

fn run(f: impl FnOnce() -> AiResult<CliOutput>) -> CliOutput {
    f().unwrap_or_else(Into::into)
}

/// `hologram ai download <repo> --revision <sha> [--offline] [--json]`
pub struct DownloadArgs {
    pub repository: String,
    pub revision: String,
    pub cache_dir: PathBuf,
    pub offline: bool,
    pub json: bool,
}

/// Download and validate an immutable HF revision. Never compiles.
pub fn download(args: DownloadArgs) -> CliOutput {
    run(|| {
        let provider = HuggingFaceProvider::new(&args.cache_dir).with_offline(args.offline);
        let request = SourceRequest::hf_repo(
            &args.repository,
            hologram_ai_huggingface::Revision::Pinned(args.revision.clone()),
        );
        let acquired = provider.acquire(
            &request,
            &mut hologram_ai_core::NullProgressSink,
            &hologram_ai_core::CancellationToken::new(),
        )?;
        let text = if args.json {
            let mut s = String::from("{");
            json_kv(&mut s, "repository", &args.repository);
            if let Some(rev) = &acquired.resolved_revision {
                json_kv_sep(&mut s, "resolvedRevision", rev);
            }
            if let Some(key) = &acquired.cache_key {
                json_kv_sep(&mut s, "cacheKey", key);
            }
            json_bool_sep(&mut s, "fromCache", acquired.from_cache);
            s.push('}');
            s
        } else {
            format!(
                "downloaded {} @ {}\ncache: {}\nfrom cache: {}",
                args.repository,
                acquired.resolved_revision.as_deref().unwrap_or(&args.revision),
                acquired.source_dir.display(),
                acquired.from_cache,
            )
        };
        Ok(CliOutput::ok(text))
    })
}

/// `hologram ai compile [repo] [--revision sha | --source dir] --output f.holo`
pub struct CompileArgs {
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub source: Option<PathBuf>,
    pub output: PathBuf,
    pub entry: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub work_dir: Option<PathBuf>,
    pub offline: bool,
    pub json: bool,
}

/// Compile through uor-r4 and write exactly one `.holo`.
pub fn compile(args: CompileArgs) -> CliOutput {
    run(|| {
        let source = match (&args.repository, &args.source) {
            (Some(repo), None) => Source::from(HuggingFaceSource::pinned(
                repo.clone(),
                args.revision.clone().unwrap_or_default(),
            )),
            (None, Some(dir)) => Source::from(LocalSource::new(dir.clone())),
            _ => {
                return Err(AiError::invalid_argument(
                    "give exactly one of a repository or --source",
                ))
            }
        };
        let mut builder = Compiler::builder()
            .source(source)
            .entry(args.entry.clone().unwrap_or_else(|| DEFAULT_ENTRY.into()))
            .offline(args.offline);
        if let Some(dir) = &args.cache_dir {
            builder = builder.cache_dir(dir);
        }
        if let Some(dir) = &args.work_dir {
            builder = builder.work_dir(dir);
        }
        let compiled = builder.build()?.compile_to_path(&args.output)?;
        let text = if args.json {
            let mut s = String::from("{");
            json_kv(&mut s, "output", &args.output.display().to_string());
            json_kv_sep(&mut s, "entry", &compiled.entry);
            json_kv_sep(&mut s, "archiveFingerprint", &hex(&compiled.archive_fingerprint));
            if let Some(rev) = &compiled.source_revision {
                json_kv_sep(&mut s, "sourceRevision", rev);
            }
            s.push('}');
            s
        } else {
            format!(
                "compiled {} -> {}\nentry: {}\nfingerprint: {}",
                compiled
                    .source_repository
                    .as_deref()
                    .unwrap_or("<local source>"),
                args.output.display(),
                compiled.entry,
                hex(&compiled.archive_fingerprint),
            )
        };
        Ok(CliOutput::ok(text))
    })
}

/// `hologram ai inspect <file.holo> [--json]`
pub struct InspectArgs {
    pub archive: PathBuf,
    pub json: bool,
}

/// Verify the archive and list model services. Never initializes an
/// inference engine.
pub fn inspect(args: InspectArgs) -> CliOutput {
    run(|| {
        let app = Application::open_path(&args.archive)?;
        let mut text = String::new();
        if args.json {
            text.push_str("{\"archiveFingerprint\":\"");
            text.push_str(&hex(&app.archive_fingerprint()));
            text.push_str("\",\"models\":[");
            for (i, m) in app.models().iter().enumerate() {
                if i > 0 {
                    text.push(',');
                }
                text.push('{');
                json_kv(&mut text, "entry", m.entry());
                json_kv_sep(&mut text, "engine", m.engine());
                json_kv_sep(&mut text, "contentKappa", m.content_kappa());
                json_kv_sep(&mut text, "artifactFormat", &m.manifest().artifact_format);
                json_kv_sep(&mut text, "modelName", &m.manifest().model_name);
                let ops: Vec<String> = m
                    .operations()
                    .iter()
                    .map(|o| format!("\"{}\"", o.name))
                    .collect();
                let _ = write!(text, ",\"operations\":[{}]", ops.join(","));
                if let Some(rev) = &m.manifest().provenance.source_revision {
                    json_kv_sep(&mut text, "sourceRevision", rev);
                }
                text.push('}');
            }
            text.push_str("]}");
        } else {
            let _ = writeln!(text, "archive fingerprint: {}", hex(&app.archive_fingerprint()));
            let _ = writeln!(text, "model services: {}", app.models().len());
            for m in app.models() {
                let _ = writeln!(
                    text,
                    "  {} (engine {}, format {})",
                    m.entry(),
                    m.engine(),
                    m.manifest().artifact_format
                );
                for op in m.operations() {
                    let modalities: Vec<String> = op
                        .inputs
                        .iter()
                        .map(|v| format!("{:?}", v.kind))
                        .collect();
                    let _ = writeln!(
                        text,
                        "    op {} [{}] -> streaming {}",
                        op.name,
                        modalities.join(","),
                        op.streaming
                    );
                }
                let p = &m.manifest().provenance;
                if let (Some(repo), Some(rev)) = (&p.source_repository, &p.source_revision) {
                    let _ = writeln!(text, "    source: {repo} @ {rev}");
                }
            }
        }
        Ok(CliOutput::ok(text))
    })
}

/// `hologram ai infer <file.holo> [--model entry] [--operation op] …`
pub struct InferArgs {
    pub archive: PathBuf,
    pub model: Option<String>,
    pub operation: Option<String>,
    pub prompt: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub json: bool,
}

/// Run one inference. Never downloads or recompiles.
pub fn infer(args: InferArgs) -> CliOutput {
    run(|| {
        let app = Application::open_path(&args.archive)?;
        let model = match &args.model {
            Some(entry) => app.model(entry)?,
            None => app.default_model()?,
        };
        let mut session = model.session()?;
        let operation = args.operation.as_deref().unwrap_or("generate");
        let mut builder = InferenceRequest::builder();
        if let Some(prompt) = &args.prompt {
            builder = builder.text("prompt", prompt);
        }
        if let Some(max) = args.max_output_tokens {
            builder = builder.max_output_tokens(max);
        }
        let completion = session.invoke(operation, builder.build())?;
        let text_out = completion.output.text("text").unwrap_or_default();
        let text = if args.json {
            let mut s = String::from("{");
            json_kv(&mut s, "text", text_out);
            json_kv_sep(&mut s, "finishReason", finish_reason_name(completion.finish_reason));
            if let Some(status) = completion.status {
                json_kv_sep(&mut s, "status", status_name(status));
            }
            json_bool_sep(&mut s, "widened", completion.widened);
            s.push('}');
            s
        } else {
            let mut s = String::new();
            if completion.finish_reason == FinishReason::Abstained {
                s.push_str("<abstained>\n");
            } else {
                let _ = writeln!(s, "{text_out}");
            }
            let _ = write!(
                s,
                "finish: {} | widened: {}",
                finish_reason_name(completion.finish_reason),
                completion.widened
            );
            if let Some(status) = completion.status {
                let _ = write!(s, " | status: {}", status_name(status));
            }
            s
        };
        Ok(CliOutput::ok(text))
    })
}

fn finish_reason_name(r: FinishReason) -> &'static str {
    match r {
        FinishReason::EndOfSequence => "end-of-sequence",
        FinishReason::OutputLimit => "output-limit",
        FinishReason::Abstained => "abstained",
        FinishReason::Cancelled => "cancelled",
    }
}

fn status_name(s: hologram_ai_core::ResolutionStatus) -> &'static str {
    use hologram_ai_core::ResolutionStatus as R;
    match s {
        R::Exact => "exact",
        R::Graph => "graph",
        R::Novel => "novel",
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn json_escape(out: &mut String, value: &str) {
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

fn json_kv(out: &mut String, key: &str, value: &str) {
    let _ = write!(out, "\"{key}\":\"");
    json_escape(out, value);
    out.push('"');
}

fn json_kv_sep(out: &mut String, key: &str, value: &str) {
    out.push(',');
    json_kv(out, key, value);
}

fn json_bool_sep(out: &mut String, key: &str, value: bool) {
    let _ = write!(out, ",\"{key}\":{value}");
}
