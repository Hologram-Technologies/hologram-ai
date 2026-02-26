//! Cast operation - convert tensor data types.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, DType, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::{dtypes, proto};

/// ONNX Cast operation.
pub struct CastOp;

impl OpTranslator for CastOp {
    fn op_type(&self) -> &'static str {
        "Cast"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;

        let is_const = ctx.is_constant(input_name);
        if !is_const {
            tracing::debug!(
                "Cast '{}': input={} is not constant",
                node.output.first().unwrap_or(&String::new()),
                input_name
            );
            return None;
        }

        let input_node = ctx.get_node(input_name)?;
        let input_const = ctx.get_constant_data(input_name)?;

        let to_dtype = get_target_dtype(node)?;
        let result = cast_constant(input_const, to_dtype)?;

        tracing::debug!(
            "Cast '{}': constant-folded {:?} -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            input_node.dtype,
            to_dtype
        );

        Some(TranslateResult::constant(
            input_node.shape.clone(),
            to_dtype,
            result,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Cast has no input")?;
        let input_node = ctx.get_node(input_name).context("Cast input not found")?;

        let to_dtype = get_target_dtype(node).context("Cast missing 'to' attribute")?;

        Ok(TranslateResult::runtime(
            OpKind::Cast { to: to_dtype },
            input_node.shape.clone(),
            to_dtype,
        ))
    }
}

fn get_target_dtype(node: &proto::NodeProto) -> Option<DType> {
    let to_value = node
        .attribute
        .iter()
        .find(|a| a.name == "to")
        .map(|a| a.i as i32)?;

    dtypes::from_onnx(to_value).ok()
}

fn cast_constant(data: &ConstantData, to: DType) -> Option<ConstantData> {
    match to {
        DType::F32 => Some(ConstantData::F32(to_f32_vec(data)?)),
        DType::F64 => Some(ConstantData::F64(to_f64_vec(data)?)),
        DType::I32 => Some(ConstantData::I32(to_i32_vec(data)?)),
        DType::I64 => Some(ConstantData::I64(to_i64_vec(data)?)),
        DType::U8 => Some(ConstantData::U8(to_u8_vec(data)?)),
        DType::Bool => Some(ConstantData::Bool(to_bool_vec(data)?)),
        _ => None, // Unsupported target type
    }
}

fn to_f32_vec(data: &ConstantData) -> Option<Vec<f32>> {
    Some(match data {
        ConstantData::F32(v) => v.clone(),
        ConstantData::F64(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::I32(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::U8(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::U16(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x as f32).collect(),
        ConstantData::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
    })
}

fn to_f64_vec(data: &ConstantData) -> Option<Vec<f64>> {
    Some(match data {
        ConstantData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::F64(v) => v.clone(),
        ConstantData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::U8(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::U16(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
    })
}

fn to_i32_vec(data: &ConstantData) -> Option<Vec<i32>> {
    Some(match data {
        ConstantData::F32(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::F64(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::I32(v) => v.clone(),
        ConstantData::I64(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::U8(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::U16(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x as i32).collect(),
        ConstantData::Bool(v) => v.iter().map(|&x| if x { 1 } else { 0 }).collect(),
    })
}

fn to_i64_vec(data: &ConstantData) -> Option<Vec<i64>> {
    Some(match data {
        ConstantData::F32(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::F64(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::I64(v) => v.clone(),
        ConstantData::U8(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::U16(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x as i64).collect(),
        ConstantData::Bool(v) => v.iter().map(|&x| if x { 1 } else { 0 }).collect(),
    })
}

fn to_u8_vec(data: &ConstantData) -> Option<Vec<u8>> {
    Some(match data {
        ConstantData::F32(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::F64(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::I32(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::U8(v) => v.clone(),
        ConstantData::U16(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x as u8).collect(),
        ConstantData::Bool(v) => v.iter().map(|&x| if x { 1 } else { 0 }).collect(),
    })
}

fn to_bool_vec(data: &ConstantData) -> Option<Vec<bool>> {
    Some(match data {
        ConstantData::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
        ConstantData::F64(v) => v.iter().map(|&x| x != 0.0).collect(),
        ConstantData::I32(v) => v.iter().map(|&x| x != 0).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x != 0).collect(),
        ConstantData::U8(v) => v.iter().map(|&x| x != 0).collect(),
        ConstantData::U16(v) => v.iter().map(|&x| x != 0).collect(),
        ConstantData::U32(v) => v.iter().map(|&x| x != 0).collect(),
        ConstantData::Bool(v) => v.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_i64_to_f32() {
        let data = ConstantData::I64(vec![1, 2, 3]);
        let result = cast_constant(&data, DType::F32).unwrap();

        if let ConstantData::F32(v) = result {
            assert_eq!(v, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Expected F32");
        }
    }

    #[test]
    fn test_cast_f32_to_i64() {
        let data = ConstantData::F32(vec![1.5, 2.7, 3.1]);
        let result = cast_constant(&data, DType::I64).unwrap();

        if let ConstantData::I64(v) = result {
            assert_eq!(v, vec![1, 2, 3]);
        } else {
            panic!("Expected I64");
        }
    }
}
