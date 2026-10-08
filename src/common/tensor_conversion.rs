//! Conversions between `tch` tensors and `ndarray` arrays.
//!
//! This bridge is used at the boundary between the LibTorch-based models and the
//! ONNX (ort) path, which operates on `ndarray` arrays end-to-end. It performs
//! explicit data copies (the conversion helpers from `tch` target an older
//! `ndarray` major version than the one required by `ort`).
//!
//! The tensor-to-array direction is the permanent seam: the Torch generators
//! convert their `Tensor` inputs/outputs once when calling the backend-neutral
//! pipeline/generation code. The array-to-tensor direction is temporary and will
//! be removed once the pipelines post-processing operates on arrays directly.

use crate::RustBertError;
use ndarray::{ArrayD, IxDyn};
use std::convert::TryInto;
use tch::{Device, Kind, Tensor};

fn tensor_shape(tensor: &Tensor) -> Vec<usize> {
    tensor
        .size()
        .iter()
        .map(|&dim| dim.max(0) as usize)
        .collect()
}

/// Convert a `tch` tensor to a dynamically dimensioned `ndarray` array of `i64`.
/// The tensor data is copied to the CPU (if needed) and cast to `i64`.
pub(crate) fn tensor_to_array_i64(tensor: &Tensor) -> Result<ArrayD<i64>, RustBertError> {
    let shape = tensor_shape(tensor);
    let flat = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .contiguous()
        .flatten(0, -1);
    let data: Vec<i64> = (&flat).try_into()?;
    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|err| {
        RustBertError::ValueError(format!("Could not convert tensor to ndarray: {err}"))
    })
}

/// Convert a `tch` tensor to a dynamically dimensioned `ndarray` array of `f32`.
/// The tensor data is copied to the CPU (if needed) and cast to `f32`.
pub(crate) fn tensor_to_array_f32(tensor: &Tensor) -> Result<ArrayD<f32>, RustBertError> {
    let shape = tensor_shape(tensor);
    let flat = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .flatten(0, -1);
    let data: Vec<f32> = (&flat).try_into()?;
    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|err| {
        RustBertError::ValueError(format!("Could not convert tensor to ndarray: {err}"))
    })
}

/// Extract the flattened data of a `tch` tensor as a vector of `i64` values.
#[allow(dead_code)]
pub(crate) fn tensor_to_vec_i64(tensor: &Tensor) -> Result<Vec<i64>, RustBertError> {
    let flat = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .contiguous()
        .flatten(0, -1);
    Ok((&flat).try_into()?)
}

/// Convert a dynamically dimensioned `ndarray` array of `f32` values to a
/// CPU-resident `tch` tensor.
pub(crate) fn array_to_tensor_f32(array: &ArrayD<f32>) -> Result<Tensor, RustBertError> {
    let shape: Vec<i64> = array.shape().iter().map(|&dim| dim as i64).collect();
    let data: Vec<f32> = array.iter().copied().collect();
    Ok(Tensor::from_slice(&data).view(shape.as_slice()))
}

/// Convert a dynamically dimensioned `ndarray` array of `i64` values to a
/// CPU-resident `tch` tensor.
pub(crate) fn array_to_tensor_i64(array: &ArrayD<i64>) -> Result<Tensor, RustBertError> {
    let shape: Vec<i64> = array.shape().iter().map(|&dim| dim as i64).collect();
    let data: Vec<i64> = array.iter().copied().collect();
    Ok(Tensor::from_slice(&data).view(shape.as_slice()))
}
