pub mod const_dedup;
pub mod const_eval;
pub mod constant_fold;
pub mod data_prop;
pub mod dead_node;
pub mod pipeline;
pub mod shape_heal;
pub mod shape_prop;
pub mod slice_to_gather;

pub use const_dedup::ConstantDeduplication;
pub use pipeline::{OptPipeline, Pass};
pub use shape_heal::ShapeHealing;
pub use shape_prop::AggressiveShapePropagation;
pub use slice_to_gather::SliceToGather;
