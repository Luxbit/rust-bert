//! Configuration and pre-trained resource definitions for the FNet model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # FNet Pretrained model weight files
pub struct FNetModelResources;

/// # FNet Pretrained model config files
pub struct FNetConfigResources;

/// # FNet Pretrained model vocab files
pub struct FNetVocabResources;

impl FNetModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/model",
        "https://huggingface.co/google/fnet-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/model",
        "https://huggingface.co/gchhablani/fnet-base-finetuned-sst2/resolve/main/rust_model.ot",
    );
}

impl FNetConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/config",
        "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/config",
        "https://huggingface.co/gchhablani/fnet-base-finetuned-sst2/resolve/main/config.json",
    );
}

impl FNetVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/google-research/tree/master/f_net>. Modified with conversion to C-array format.
    pub const BASE: (&'static str, &'static str) = (
        "fnet-base/spiece",
        "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/gchhablani/fnet-base-finetuned-sst2>. Modified with conversion to C-array format.
    pub const BASE_SST2: (&'static str, &'static str) = (
        "fnet-base-sst2/spiece",
        "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize)]
/// # FNet model configuration
/// Defines the FNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct FNetConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub intermediate_size: i64,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub initializer_range: f64,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub decoder_start_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config for FNetConfig {}

impl Default for FNetConfig {
    fn default() -> Self {
        FNetConfig {
            vocab_size: 32000,
            hidden_size: 768,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            hidden_act: Activation::gelu_new,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 4,
            initializer_range: 0.02,
            layer_norm_eps: Some(1e-12),
            pad_token_id: Some(3),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            decoder_start_token_id: None,
            id2label: None,
            label2id: None,
            output_attentions: None,
            output_hidden_states: None,
        }
    }
}
