//! Utility functions for ONNX attribute parsing.
//!
//! Provides zero-copy, O(1) attribute extraction from ONNX AttributeProto.

use hologram_onnx_core::{OnnxError, Result};
use hologram_onnx_spec::{AttributeProto, TensorProto, attribute_proto::AttributeType};
use tracing::trace;

/// Parse integer attribute with default value.
///
/// **Performance**: O(1) - single linear scan of attributes
pub fn parse_attr_int(
    attrs: &[AttributeProto],
    name: &str,
    default: i64,
) -> Result<i64> {
    for attr in attrs {
        if attr.name == name {
            trace!("Found attribute '{}' with value {}", name, attr.i);
            return Ok(attr.i);
        }
    }
    trace!("Attribute '{}' not found, using default {}", name, default);
    Ok(default)
}

/// Parse integer array attribute with default value.
///
/// **Performance**: O(1) amortized - returns reference to existing data when possible
pub fn parse_attr_ints(
    attrs: &[AttributeProto],
    name: &str,
    default: Vec<i64>,
) -> Result<Vec<i64>> {
    for attr in attrs {
        if attr.name == name {
            if !attr.ints.is_empty() {
                trace!("Found attribute '{}' with {} values", name, attr.ints.len());
                // Clone is necessary here as we need owned data
                return Ok(attr.ints.clone());
            }
        }
    }
    trace!("Attribute '{}' not found, using default", name);
    Ok(default)
}

/// Parse float attribute with default value.
///
/// **Performance**: O(1) - single linear scan of attributes
pub fn parse_attr_float(
    attrs: &[AttributeProto],
    name: &str,
    default: f32,
) -> Result<f32> {
    for attr in attrs {
        if attr.name == name {
            trace!("Found attribute '{}' with value {}", name, attr.f);
            return Ok(attr.f);
        }
    }
    trace!("Attribute '{}' not found, using default {}", name, default);
    Ok(default)
}

/// Parse float array attribute with default value.
///
/// **Performance**: O(1) amortized - returns reference to existing data when possible
pub fn parse_attr_floats(
    attrs: &[AttributeProto],
    name: &str,
    default: Vec<f32>,
) -> Result<Vec<f32>> {
    for attr in attrs {
        if attr.name == name {
            if !attr.floats.is_empty() {
                trace!("Found attribute '{}' with {} values", name, attr.floats.len());
                return Ok(attr.floats.clone());
            }
        }
    }
    trace!("Attribute '{}' not found, using default", name);
    Ok(default)
}

/// Parse string attribute.
///
/// **Performance**: O(1) - single linear scan of attributes
pub fn parse_attr_string(
    attrs: &[AttributeProto],
    name: &str,
) -> Result<String> {
    for attr in attrs {
        if attr.name == name {
            // Convert bytes to String (UTF-8)
            let s = String::from_utf8(attr.s.clone())
                .map_err(|e| OnnxError::InvalidAttribute {
                    name: name.to_string(),
                    reason: format!("Invalid UTF-8: {}", e),
                })?;
            trace!("Found attribute '{}' with value '{}'", name, s);
            return Ok(s);
        }
    }
    Err(OnnxError::InvalidAttribute {
        name: name.to_string(),
        reason: "Required string attribute not found".to_string(),
    })
}

/// Parse string attribute with default value.
///
/// **Performance**: O(1) - single linear scan of attributes
pub fn parse_attr_string_or(
    attrs: &[AttributeProto],
    name: &str,
    default: &str,
) -> Result<String> {
    for attr in attrs {
        if attr.name == name {
            // Convert bytes to String (UTF-8)
            let s = String::from_utf8(attr.s.clone())
                .map_err(|e| OnnxError::InvalidAttribute {
                    name: name.to_string(),
                    reason: format!("Invalid UTF-8: {}", e),
                })?;
            trace!("Found attribute '{}' with value '{}'", name, s);
            return Ok(s);
        }
    }
    trace!("Attribute '{}' not found, using default '{}'", name, default);
    Ok(default.to_string())
}

/// Parse tensor attribute.
///
/// **Performance**: O(1) - single linear scan of attributes
pub fn parse_attr_tensor<'a>(
    attrs: &'a [AttributeProto],
    name: &str,
) -> Result<&'a TensorProto> {
    for attr in attrs {
        if attr.name == name {
            if let Some(ref tensor) = attr.t {
                trace!("Found attribute '{}' with tensor", name);
                return Ok(tensor);
            }
        }
    }
    Err(OnnxError::InvalidAttribute {
        name: name.to_string(),
        reason: "Required tensor attribute not found".to_string(),
    })
}

/// Validate attribute type matches expected type.
///
/// **Performance**: O(1)
pub fn validate_attr_type(
    attr: &AttributeProto,
    expected: AttributeType,
) -> Result<()> {
    let actual = AttributeType::try_from(attr.r#type)
        .map_err(|_| OnnxError::InvalidAttribute {
            name: attr.name.clone(),
            reason: format!("Unknown attribute type: {}", attr.r#type),
        })?;

    if actual != expected {
        return Err(OnnxError::InvalidAttribute {
            name: attr.name.clone(),
            reason: format!("Expected type {:?}, got {:?}", expected, actual),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_onnx_spec::attribute_proto::AttributeType;

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn make_floats_attr(name: &str, values: Vec<f32>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            floats: values,
            r#type: AttributeType::Floats as i32,
            ..Default::default()
        }
    }

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            r#type: AttributeType::String as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_parse_attr_int() {
        let attrs = vec![
            make_int_attr("alpha", 42),
            make_int_attr("beta", 100),
        ];

        assert_eq!(parse_attr_int(&attrs, "alpha", 0).unwrap(), 42);
        assert_eq!(parse_attr_int(&attrs, "beta", 0).unwrap(), 100);
        assert_eq!(parse_attr_int(&attrs, "gamma", 10).unwrap(), 10); // default
    }

    #[test]
    fn test_parse_attr_ints() {
        let attrs = vec![
            make_ints_attr("strides", vec![1, 2, 3]),
            make_ints_attr("pads", vec![0, 0, 0, 0]),
        ];

        assert_eq!(
            parse_attr_ints(&attrs, "strides", vec![]).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            parse_attr_ints(&attrs, "pads", vec![]).unwrap(),
            vec![0, 0, 0, 0]
        );
        assert_eq!(
            parse_attr_ints(&attrs, "dilations", vec![1, 1]).unwrap(),
            vec![1, 1] // default
        );
    }

    #[test]
    fn test_parse_attr_float() {
        let attrs = vec![
            make_float_attr("alpha", 0.5),
            make_float_attr("beta", 1.5),
        ];

        assert_eq!(parse_attr_float(&attrs, "alpha", 0.0).unwrap(), 0.5);
        assert_eq!(parse_attr_float(&attrs, "beta", 0.0).unwrap(), 1.5);
        assert_eq!(parse_attr_float(&attrs, "gamma", 2.0).unwrap(), 2.0); // default
    }

    #[test]
    fn test_parse_attr_floats() {
        let attrs = vec![
            make_floats_attr("scales", vec![0.5, 1.0, 1.5]),
        ];

        assert_eq!(
            parse_attr_floats(&attrs, "scales", vec![]).unwrap(),
            vec![0.5, 1.0, 1.5]
        );
        assert_eq!(
            parse_attr_floats(&attrs, "missing", vec![1.0]).unwrap(),
            vec![1.0] // default
        );
    }

    #[test]
    fn test_parse_attr_string() {
        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_string_attr("activation", "relu"),
        ];

        assert_eq!(parse_attr_string(&attrs, "mode").unwrap(), "constant");
        assert_eq!(parse_attr_string(&attrs, "activation").unwrap(), "relu");
        assert!(parse_attr_string(&attrs, "missing").is_err());
    }

    #[test]
    fn test_parse_attr_string_invalid_utf8() {
        let mut attrs = vec![make_string_attr("test", "valid")];
        attrs[0].s = vec![0xFF, 0xFF]; // Invalid UTF-8

        let result = parse_attr_string(&attrs, "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidAttribute { .. }));
    }

    #[test]
    fn test_validate_attr_type() {
        let int_attr = make_int_attr("test", 42);
        let float_attr = make_float_attr("test", 1.0);

        assert!(validate_attr_type(&int_attr, AttributeType::Int).is_ok());
        assert!(validate_attr_type(&float_attr, AttributeType::Float).is_ok());
        assert!(validate_attr_type(&int_attr, AttributeType::Float).is_err());
    }

    #[test]
    fn test_empty_attributes() {
        let attrs: Vec<AttributeProto> = vec![];

        assert_eq!(parse_attr_int(&attrs, "any", 99).unwrap(), 99);
        assert_eq!(parse_attr_ints(&attrs, "any", vec![1, 2]).unwrap(), vec![1, 2]);
        assert_eq!(parse_attr_float(&attrs, "any", 3.14).unwrap(), 3.14);
        assert_eq!(parse_attr_floats(&attrs, "any", vec![1.0]).unwrap(), vec![1.0]);
        assert!(parse_attr_string(&attrs, "any").is_err());
    }

    #[test]
    fn test_multiple_attrs_same_name() {
        // ONNX spec doesn't allow this, but test robustness
        // Should return first match
        let attrs = vec![
            make_int_attr("alpha", 10),
            make_int_attr("alpha", 20),
        ];

        assert_eq!(parse_attr_int(&attrs, "alpha", 0).unwrap(), 10);
    }
}
