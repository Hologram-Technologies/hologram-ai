//! Per-op-category tolerance definitions for numerical comparison.
//!
//! Uses numpy `allclose` semantics: `|actual - expected| <= atol + rtol * |expected|`

/// Absolute and relative tolerance for comparing kernel outputs.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub atol: f32,
    pub rtol: f32,
}

impl Tolerance {
    /// Check if two values are close within this tolerance.
    pub fn is_close(&self, actual: f32, expected: f32) -> bool {
        if actual.is_nan() && expected.is_nan() {
            return true;
        }
        if actual.is_nan() || expected.is_nan() {
            return false;
        }
        (actual - expected).abs() <= self.atol + self.rtol * expected.abs()
    }
}

/// Get the appropriate tolerance for a canonical op, keyed by its `OpKind`
/// name (`hologram_ops::OpKind::name()`). Compute-heavy ops (matmul, conv,
/// attention) accumulate more float error, so they get looser bounds; exact
/// integer/boolean ops get zero tolerance.
pub fn tolerance_for(op_name: &str) -> Tolerance {
    match op_name {
        // Exact: comparisons, boolean, integer-index ops.
        "Equal" | "Less" | "LessOrEqual" | "Greater" | "GreaterOrEqual" | "And" | "Or" | "Xor"
        | "Bnot" | "Sign" | "IsNaN" => Tolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        // Accumulating linear algebra / normalization.
        "MatMul" | "Gemm" | "Conv2d" | "ConvTranspose2d" | "RmsNorm" | "LayerNorm"
        | "GroupNorm" | "InstanceNorm" | "AddRmsNorm" | "FusedSwiGlu" => Tolerance {
            atol: 1e-4,
            rtol: 1e-3,
        },
        "Softmax" | "LogSoftmax" => Tolerance {
            atol: 1e-5,
            rtol: 1e-4,
        },
        "Attention" => Tolerance {
            atol: 1e-3,
            rtol: 1e-2,
        },
        // Default: elementwise math.
        _ => Tolerance {
            atol: 1e-6,
            rtol: 1e-5,
        },
    }
}

/// Compare two f32 slices and return a detailed comparison result.
pub fn compare_outputs(actual: &[f32], expected: &[f32], tol: Tolerance) -> ComparisonResult {
    if actual.len() != expected.len() {
        return ComparisonResult {
            passed: false,
            max_abs_error: f32::INFINITY,
            max_rel_error: f32::INFINITY,
            worst_index: 0,
            num_mismatches: actual.len().max(expected.len()),
            total_elements: actual.len().max(expected.len()),
            message: format!(
                "length mismatch: actual={} expected={}",
                actual.len(),
                expected.len()
            ),
        };
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_idx = 0;
    let mut mismatches = 0;

    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if !tol.is_close(*a, *e) {
            mismatches += 1;
        }
        let abs_err = (a - e).abs();
        let rel_err = if *e != 0.0 {
            abs_err / e.abs()
        } else {
            abs_err
        };
        if abs_err > max_abs {
            max_abs = abs_err;
            max_rel = rel_err;
            worst_idx = i;
        }
    }

    let passed = mismatches == 0;
    let message = if passed {
        format!(
            "PASS (max_abs={max_abs:.2e}, {}/{} elements)",
            actual.len(),
            actual.len()
        )
    } else {
        format!(
            "FAIL: {mismatches}/{} elements exceed tolerance (atol={}, rtol={})\n  worst: idx={worst_idx} actual={} expected={} abs_err={max_abs:.2e}",
            actual.len(),
            tol.atol,
            tol.rtol,
            actual[worst_idx],
            expected[worst_idx],
        )
    };

    ComparisonResult {
        passed,
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        worst_index: worst_idx,
        num_mismatches: mismatches,
        total_elements: actual.len(),
        message,
    }
}

/// Result of comparing two tensors.
#[derive(Debug)]
pub struct ComparisonResult {
    pub passed: bool,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub worst_index: usize,
    pub num_mismatches: usize,
    pub total_elements: usize,
    pub message: String,
}
