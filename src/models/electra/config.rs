//! Configuration and pre-trained resource definitions for the Electra model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # Electra Pretrained model weight files
pub struct ElectraModelResources;

/// # Electra Pretrained model config files
pub struct ElectraConfigResources;

/// # Electra Pretrained model vocab files
pub struct ElectraVocabResources;

impl ElectraModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/model",
        "https://huggingface.co/google/electra-base-generator/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/model",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/rust_model.ot",
    );
}

impl ElectraConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/config",
        "https://huggingface.co/google/electra-base-generator/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/config",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/config.json",
    );
}

impl ElectraVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_GENERATOR: (&'static str, &'static str) = (
        "electra-base-generator/vocab",
        "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/electra>. Modified with conversion to C-array format.
    pub const BASE_DISCRIMINATOR: (&'static str, &'static str) = (
        "electra-base-discriminator/vocab",
        "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Electra model configuration
/// Defines the Electra model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct ElectraConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub pad_token_id: i64,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for ElectraConfig {}

impl Default for ElectraConfig {
    fn default() -> Self {
        ElectraConfig {
            hidden_act: Activation::gelu,
            attention_probs_dropout_prob: 0.1,
            embedding_size: 128,
            hidden_dropout_prob: 0.1,
            hidden_size: 256,
            initializer_range: 0.02,
            layer_norm_eps: Some(1e-12),
            intermediate_size: 1024,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30522,
            pad_token_id: 0,
            output_past: None,
            output_attentions: None,
            output_hidden_states: None,
            id2label: None,
            label2id: None,
        }
    }
}
