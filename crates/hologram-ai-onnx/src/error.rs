use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),

    #[error("missing graph in model proto")]
    NoGraph,

    #[error("import failed: {0}")]
    Import(#[from] anyhow::Error),
}
