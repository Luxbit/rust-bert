//! Conversions between `ndarray` arrays and `ort` values, used by the ONNX sessions.
use crate::RustBertError;
use ndarray::ArrayD;
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::{DynValue, Tensor};

/// Input for an ONNX session, holding an owned array of a supported element type.
pub(crate) enum ONNXInput {
    I64(ArrayD<i64>),
    F32(ArrayD<f32>),
}

impl ONNXInput {
    pub(crate) fn into_value(self) -> Result<SessionInputValue<'static>, RustBertError> {
        match self {
            ONNXInput::I64(array) => Ok(Tensor::from_array(array)?.into_dyn().into()),
            ONNXInput::F32(array) => Ok(Tensor::from_array(array)?.into_dyn().into()),
        }
    }
}

/// Extract an `ort` output tensor as a dynamically dimensioned `ndarray` of `f32` values.
///
/// `f32` outputs are extracted directly; `f16`/`bf16` outputs are converted to `f32` on the fly.
pub(crate) fn ort_output_to_array_f32(value: &DynValue) -> Result<ArrayD<f32>, RustBertError> {
    if let Ok(view) = value.try_extract_array::<f32>() {
        return Ok(view.to_owned());
    }
    if let Ok(view) = value.try_extract_array::<half::f16>() {
        return Ok(view.mapv(half::f16::to_f32));
    }
    if let Ok(view) = value.try_extract_array::<half::bf16>() {
        return Ok(view.mapv(half::bf16::to_f32));
    }
    Err(RustBertError::OrtError(
        "ONNX output could not be extracted: expected a f32 tensor.".to_string(),
    ))
}

/// Helper to collect the key/value cached states from the outputs of a decoder session.
pub(crate) fn key_values_from_outputs(
    outputs: &SessionOutputs,
    key_value_names: &std::collections::HashMap<String, usize>,
) -> Result<std::collections::HashMap<String, ArrayD<f32>>, RustBertError> {
    key_value_names
        .iter()
        .filter(|(name, _)| name.contains("key") | name.contains("value"))
        .map(|(name, _)| {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| RustBertError::OrtError(format!("Output {name} not found.")))?;
            Ok((name.clone(), ort_output_to_array_f32(value)?))
        })
        .collect()
}
