//! Backend-neutral tensor operations on `ndarray` arrays.
//!
//! These operations replicate the subset of tensor arithmetic required by the
//! pipelines post-processing and the text-generation loop when running on the
//! ONNX (`ort`) backend, where logits and cached states are handled as
//! `ndarray` arrays end-to-end.

use ndarray::{ArrayD, Axis, IxDyn};

/// Compute the softmax over the last dimension of an array.
#[allow(dead_code)] // used by the generation pipelines from phase 2 onwards
pub fn softmax_last_dim(array: &ArrayD<f32>) -> ArrayD<f32> {
    let axis = array.ndim() - 1;
    let max = array.map_axis(Axis(axis), |lane| {
        lane.fold(f32::NEG_INFINITY, |acc, &value| acc.max(value))
    });
    let exp = (array - &max.insert_axis(Axis(axis))).mapv(f32::exp);
    let sum = exp.sum_axis(Axis(axis)).insert_axis(Axis(axis));
    exp / sum
}

/// Compute the log-softmax over the last dimension of an array.
#[allow(dead_code)] // used by the generation pipelines from phase 2 onwards
pub fn log_softmax_last_dim(array: &ArrayD<f32>) -> ArrayD<f32> {
    let softmax = softmax_last_dim(array);
    softmax.mapv(f32::ln)
}

/// Compute the element-wise sigmoid function.
#[allow(dead_code)] // used by the classification pipelines' multi-label scoring
pub fn sigmoid(array: &ArrayD<f32>) -> ArrayD<f32> {
    array.mapv(|value| 1.0 / (1.0 + (-value).exp()))
}

/// Select the top-k elements along the last dimension.
///
/// Returns the k largest values (sorted in descending order) and their indices,
/// mirroring the behavior of `tch::Tensor::topk` with `dim: -1, sorted: true`.
#[allow(dead_code)] // used by the generation pipelines from phase 2 onwards
pub fn topk_last_dim(array: &ArrayD<f32>, k: usize) -> (ArrayD<f32>, ArrayD<i64>) {
    let axis = array.ndim() - 1;
    let mut shape: Vec<usize> = array.shape().to_vec();
    let last = shape[axis];
    let k = k.min(last);
    shape[axis] = k;

    let lane_len = last;
    let num_lanes = array.len() / lane_len.max(1);
    let mut values = Vec::with_capacity(num_lanes * k);
    let mut indices = Vec::with_capacity(num_lanes * k);
    let view = array.view();
    let flat = view.to_shape(IxDyn(&[num_lanes, lane_len])).unwrap();
    for lane in flat.rows() {
        let mut order: Vec<usize> = (0..lane_len).collect();
        order.sort_unstable_by(|&a, &b| {
            lane[b]
                .partial_cmp(&lane[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(k);
        for &index in &order {
            values.push(lane[index]);
            indices.push(index as i64);
        }
    }

    (
        ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape), indices).unwrap(),
    )
}

/// Compute the argmax over the last dimension of an array.
///
/// Ties are resolved by taking the lowest index, mirroring `tch::Tensor::argmax`.
#[allow(dead_code)] // used by the generation pipelines from phase 2 onwards
pub fn argmax_last_dim(array: &ArrayD<f32>) -> ArrayD<i64> {
    let axis = array.ndim() - 1;
    array.map_axis(Axis(axis), |lane| {
        let mut best_index = 0i64;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in lane.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index as i64;
            }
        }
        best_index
    })
}

/// Gather rows along the first axis (`index_select(0, indices)`).
pub fn gather_rows<T: Clone>(array: &ArrayD<T>, indices: &[i64]) -> ArrayD<T> {
    let mut shape: Vec<usize> = array.shape().to_vec();
    let row_len = shape[1..].iter().product::<usize>().max(1);
    shape[0] = indices.len();
    let mut data = Vec::with_capacity(indices.len() * row_len);
    for &index in indices {
        data.extend(
            array
                .index_axis(Axis(0), index.max(0) as usize)
                .iter()
                .cloned(),
        );
    }
    ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
}

/// Create an array of `i64` filled with `value` and the provided shape.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub fn full_i64(shape: &[usize], value: i64) -> ArrayD<i64> {
    ArrayD::from_elem(IxDyn(shape), value)
}

/// Create an array of `i64` ones with the provided shape.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub fn ones_i64(shape: &[usize]) -> ArrayD<i64> {
    full_i64(shape, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax_last_dim() {
        let input = array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn();
        let output = softmax_last_dim(&input);
        let expected_0 = {
            let exp: [f32; 3] = [(-2.0f32).exp(), (-1.0f32).exp(), 1.0];
            let sum: f32 = exp.iter().sum();
            exp.iter().map(|&value| value / sum).collect::<Vec<f32>>()
        };
        for (actual, expected) in output.index_axis(Axis(0), 0).iter().zip(expected_0) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for &value in output.index_axis(Axis(0), 1) {
            assert!((value - 1.0 / 3.0).abs() < 1e-6);
        }
        // Rows sum to 1
        for row in output.rows() {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let input = array![[1000.0f32, 1001.0, 1002.0]].into_dyn();
        let output = softmax_last_dim(&input);
        for &value in output.iter() {
            assert!(value.is_finite());
        }
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_last_dim() {
        let input = array![[1.0f32, 2.0, 3.0]].into_dyn();
        let output = log_softmax_last_dim(&input);
        let softmax = softmax_last_dim(&input);
        for (log_p, p) in output.iter().zip(softmax.iter()) {
            assert!((log_p - p.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_topk_last_dim() {
        let input = array![[0.1f32, 0.5, 0.3, 0.9], [0.7, 0.2, 0.6, 0.4]].into_dyn();
        let (values, indices) = topk_last_dim(&input, 2);
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(indices.shape(), &[2, 2]);
        assert_eq!(
            values
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.9, 0.5]
        );
        assert_eq!(
            indices
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert_eq!(
            values
                .index_axis(Axis(0), 1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.7, 0.6]
        );
        assert_eq!(
            indices
                .index_axis(Axis(0), 1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn test_topk_k_larger_than_axis() {
        let input = array![[1.0f32, 2.0]].into_dyn();
        let (values, indices) = topk_last_dim(&input, 5);
        assert_eq!(values.shape(), &[1, 2]);
        assert_eq!(
            values
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![2.0, 1.0]
        );
        assert_eq!(
            indices
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn test_argmax_last_dim() {
        let input = array![[1.0f32, 3.0, 3.0], [2.0, 1.0, 0.0]].into_dyn();
        let output = argmax_last_dim(&input);
        assert_eq!(output.shape(), &[2]);
        // Ties resolved by lowest index
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 0);
    }

    #[test]
    fn test_gather_rows() {
        let input = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
        let output = gather_rows(&input, &[2, 0, 2]);
        assert_eq!(output.shape(), &[3, 2]);
        assert_eq!(
            output
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![5.0, 6.0]
        );
        assert_eq!(
            output
                .index_axis(Axis(0), 1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            output
                .index_axis(Axis(0), 2)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![5.0, 6.0]
        );
    }

    #[test]
    fn test_gather_rows_3d() {
        let input = array![[[1.0f32], [2.0]], [[3.0], [4.0]]].into_dyn();
        let output = gather_rows(&input, &[1, 1]);
        assert_eq!(output.shape(), &[2, 2, 1]);
        assert_eq!(output[[0, 0, 0]], 3.0);
        assert_eq!(output[[0, 1, 0]], 4.0);
        assert_eq!(output[[1, 0, 0]], 3.0);
    }

    #[test]
    fn test_full_and_ones_i64() {
        let output = full_i64(&[2, 3], 7);
        assert_eq!(output.shape(), &[2, 3]);
        assert!(output.iter().all(|&value| value == 7));
        let ones = ones_i64(&[4]);
        assert_eq!(ones.shape(), &[4]);
        assert!(ones.iter().all(|&value| value == 1));
    }
}
