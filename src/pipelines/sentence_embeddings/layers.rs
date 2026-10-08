use serde::{de, Deserialize, Deserializer};
use std::path::Path;

use crate::common::activations::Activation;
#[cfg(feature = "libtorch")]
use crate::common::activations::TensorFunction;
use crate::Config;
#[cfg(feature = "libtorch")]
use crate::RustBertError;
#[cfg(feature = "onnx")]
use ndarray::{Array2, Array3};
#[cfg(feature = "libtorch")]
use tch::{nn, Device, Kind, Tensor};

/// Configuration for [`Pooling`](Pooling) layer.
#[derive(Debug, Deserialize)]
pub struct PoolingConfig {
    /// Dimensions for the word embeddings
    pub word_embedding_dimension: i64,
    /// Use the first token (CLS token) as text representations
    pub pooling_mode_cls_token: bool,
    /// Use max in each dimension over all tokens
    pub pooling_mode_max_tokens: bool,
    /// Perform mean-pooling
    pub pooling_mode_mean_tokens: bool,
    /// Perform mean-pooling, but devide by sqrt(input_length)
    pub pooling_mode_mean_sqrt_len_tokens: bool,
}

impl Config for PoolingConfig {}

/// Performs pooling (max or mean) on the token embeddings.
///
/// Using pooling, it generates from a variable sized sentence a fixed sized sentence
/// embedding. You can concatenate multiple poolings together.
pub struct Pooling {
    conf: PoolingConfig,
}

impl Pooling {
    pub fn new(conf: PoolingConfig) -> Pooling {
        Pooling { conf }
    }

    #[cfg(feature = "libtorch")]
    pub fn forward(&self, mut token_embeddings: Tensor, attention_mask: &Tensor) -> Tensor {
        let mut output_vectors = Vec::new();

        if self.conf.pooling_mode_cls_token {
            let cls_token = token_embeddings.select(1, 0); // Take first token by default
            output_vectors.push(cls_token);
        }

        if self.conf.pooling_mode_max_tokens {
            let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(&token_embeddings);
            // Set padding tokens to large negative value
            token_embeddings = token_embeddings.masked_fill_(&input_mask_expanded.eq(0), -1e9);
            let max_over_time = token_embeddings.max_dim(1, true).0;
            output_vectors.push(max_over_time);
        }

        if self.conf.pooling_mode_mean_tokens || self.conf.pooling_mode_mean_sqrt_len_tokens {
            let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(&token_embeddings);
            let sum_embeddings = (token_embeddings * &input_mask_expanded).sum_dim_intlist(
                [1].as_slice(),
                false,
                Kind::Float,
            );
            let sum_mask = input_mask_expanded.sum_dim_intlist([1].as_slice(), false, Kind::Float);
            let sum_mask = sum_mask.clamp_min(10e-9);

            if self.conf.pooling_mode_mean_tokens {
                output_vectors.push(&sum_embeddings / &sum_mask);
            }
            if self.conf.pooling_mode_mean_sqrt_len_tokens {
                output_vectors.push(sum_embeddings / sum_mask.sqrt());
            }
        }

        Tensor::cat(&output_vectors, 1)
    }
}

/// Configuration for [`Dense`](Dense) layer.
#[derive(Debug, Deserialize)]
pub struct DenseConfig {
    /// Size of the input dimension
    pub in_features: i64,
    /// Output size
    pub out_features: i64,
    /// Add a bias vector
    pub bias: bool,
    /// Activation function applied on output
    #[serde(deserialize_with = "last_part")]
    pub activation_function: Activation,
}

impl Config for DenseConfig {}

/// Split the given string on `.` and try to construct an `Activation` from the last part
fn last_part<'de, D>(deserializer: D) -> Result<Activation, D::Error>
where
    D: Deserializer<'de>,
{
    let activation = String::deserialize(deserializer)?;
    activation
        .split('.')
        .next_back()
        .map(|s| serde_json::from_value(serde_json::Value::String(s.to_lowercase())))
        .transpose()
        .map_err(de::Error::custom)?
        .ok_or_else(|| format!("Invalid Activation: {activation}"))
        .map_err(de::Error::custom)
}

/// Feed-forward function with activiation function.
///
/// This layer takes a fixed-sized sentence embedding and passes it through a
/// feed-forward layer. Can be used to generate deep averaging networs (DAN).
#[cfg(feature = "libtorch")]
pub struct Dense {
    linear: nn::Linear,
    activation: TensorFunction,
    _var_store: nn::VarStore,
}

#[cfg(feature = "libtorch")]
impl Dense {
    pub fn new<P: AsRef<Path>>(
        dense_conf: DenseConfig,
        dense_weights: P,
        device: Device,
    ) -> Result<Dense, RustBertError> {
        let mut vs_dense = nn::VarStore::new(device);

        let linear_conf = nn::LinearConfig {
            ws_init: nn::Init::Const(0.),
            bs_init: Some(nn::Init::Const(0.)),
            bias: dense_conf.bias,
        };
        let linear = nn::linear(
            vs_dense.root(),
            dense_conf.in_features,
            dense_conf.out_features,
            linear_conf,
        );

        let activation = dense_conf.activation_function.get_function();

        vs_dense.load(dense_weights)?;

        Ok(Dense {
            linear,
            activation,
            _var_store: vs_dense,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.activation.get_fn()(&x.apply(&self.linear))
    }
}

impl Pooling {
    /// Pool token embeddings (batch, sequence, hidden) into sentence embeddings
    /// (batch, output_dim) using `ndarray`, for the ONNX backend.
    #[cfg(feature = "onnx")]
    pub fn forward_array(
        &self,
        token_embeddings: &ndarray::Array3<f32>,
        attention_mask: &ndarray::Array2<i64>,
    ) -> ndarray::Array2<f32> {
        let (batch, seq, hidden) = (
            token_embeddings.shape()[0],
            token_embeddings.shape()[1],
            token_embeddings.shape()[2],
        );
        let mut output_vectors: Vec<ndarray::Array2<f32>> = Vec::new();

        if self.conf.pooling_mode_cls_token {
            let mut cls = ndarray::Array2::<f32>::zeros((batch, hidden));
            for b in 0..batch {
                for h in 0..hidden {
                    cls[[b, h]] = token_embeddings[[b, 0, h]];
                }
            }
            output_vectors.push(cls);
        }

        if self.conf.pooling_mode_max_tokens {
            let mut max_pool =
                ndarray::Array2::<f32>::from_elem((batch, hidden), f32::NEG_INFINITY);
            for b in 0..batch {
                for s in 0..seq {
                    if attention_mask[[b, s]] == 0 {
                        continue;
                    }
                    for h in 0..hidden {
                        max_pool[[b, h]] = max_pool[[b, h]].max(token_embeddings[[b, s, h]]);
                    }
                }
            }
            output_vectors.push(max_pool);
        }

        if self.conf.pooling_mode_mean_tokens || self.conf.pooling_mode_mean_sqrt_len_tokens {
            let mut sum_embeddings = ndarray::Array2::<f32>::zeros((batch, hidden));
            let mut sum_mask = ndarray::Array2::<f32>::zeros((batch, 1));
            for b in 0..batch {
                let mut count = 0f32;
                for s in 0..seq {
                    if attention_mask[[b, s]] == 0 {
                        continue;
                    }
                    count += 1f32;
                    for h in 0..hidden {
                        sum_embeddings[[b, h]] += token_embeddings[[b, s, h]];
                    }
                }
                sum_mask[[b, 0]] = count.max(10e-9);
            }
            if self.conf.pooling_mode_mean_tokens {
                output_vectors.push(sum_embeddings.clone() / &sum_mask);
            }
            if self.conf.pooling_mode_mean_sqrt_len_tokens {
                let sqrt_mask = sum_mask.mapv(|value| value.sqrt());
                output_vectors.push(sum_embeddings / &sqrt_mask);
            }
        }

        // concatenate pooling outputs along the hidden dimension
        let total_dim: usize = output_vectors.iter().map(|v| v.ncols()).sum();
        let mut output = ndarray::Array2::<f32>::zeros((batch, total_dim));
        let mut offset = 0;
        for vector in output_vectors {
            for b in 0..batch {
                for h in 0..vector.ncols() {
                    output[[b, offset + h]] = vector[[b, h]];
                }
            }
            offset += vector.ncols();
        }
        output
    }
}
