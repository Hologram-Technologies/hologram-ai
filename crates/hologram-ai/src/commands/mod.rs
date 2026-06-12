pub mod export_fixture;
pub mod generate;
pub mod run_cmd;

use crate::compiler::ModelCompiler;
use hologram_ai_common::lower::QuantStrategy;

#[derive(Clone, Debug, Default)]
pub(crate) struct CompileCliOptions {
    pub seq_len: Option<u64>,
    pub quantize: Option<String>,
    pub spatial_scale: Option<u32>,
}

pub(crate) fn build_model_compiler(options: &CompileCliOptions) -> anyhow::Result<ModelCompiler> {
    Ok(ModelCompiler {
        mmap: true,
        seq_len_override: options.seq_len,
        quant_strategy: parse_quant(options.quantize.as_deref())?,
        spatial_scale: options.spatial_scale,
        patch_budget_ratio: Some(0.75),
        address_model: false,
    })
}

fn parse_quant(s: Option<&str>) -> anyhow::Result<QuantStrategy> {
    Ok(match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("none") | Some("f32") => QuantStrategy::None,
        Some("int8") => QuantStrategy::Int8,
        Some("int4") => QuantStrategy::Int4,
        Some(other) => {
            anyhow::bail!("unknown quantization scheme {other:?} (expected none/int8/int4)")
        }
    })
}
