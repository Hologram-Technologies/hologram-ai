//! Core comparator: runs a FloatOp through both dispatch_float and the
//! reference implementation, compares outputs.

use crate::tolerance::{compare_outputs, tolerance_for, ComparisonResult, Tolerance};
use hologram::FloatOp;
use hologram::hologram_exec::float_dispatch::dispatch_float;

/// Run a FloatOp through dispatch_float and compare against expected output.
pub fn verify_dispatch(
    op: &FloatOp,
    inputs: &[Vec<u8>],
    expected_output: &[f32],
) -> ComparisonResult {
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let tol = tolerance_for(op);

    match dispatch_float(op, &input_refs) {
        Ok(result) => {
            let actual: &[f32] = bytemuck::cast_slice(&result);
            compare_outputs(actual, expected_output, tol)
        }
        Err(e) => ComparisonResult {
            passed: false,
            max_abs_error: f32::INFINITY,
            max_rel_error: f32::INFINITY,
            worst_index: 0,
            num_mismatches: expected_output.len(),
            total_elements: expected_output.len(),
            message: format!("dispatch_float returned error: {e:?}"),
        },
    }
}

/// Run a FloatOp and compare against expected output with custom tolerance.
pub fn verify_dispatch_with_tolerance(
    op: &FloatOp,
    inputs: &[Vec<u8>],
    expected_output: &[f32],
    tol: Tolerance,
) -> ComparisonResult {
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

    match dispatch_float(op, &input_refs) {
        Ok(result) => {
            let actual: &[f32] = bytemuck::cast_slice(&result);
            compare_outputs(actual, expected_output, tol)
        }
        Err(e) => ComparisonResult {
            passed: false,
            max_abs_error: f32::INFINITY,
            max_rel_error: f32::INFINITY,
            worst_index: 0,
            num_mismatches: expected_output.len(),
            total_elements: expected_output.len(),
            message: format!("dispatch_float returned error: {e:?}"),
        },
    }
}
