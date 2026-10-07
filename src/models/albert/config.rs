//! Configuration and pre-trained resource definitions for the Albert model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # ALBERT Pretrained model weight files
pub struct AlbertModelResources;

/// # ALBERT Pretrained model config files
pub struct AlbertConfigResources;

/// # ALBERT Pretrained model vocab files
pub struct AlbertVocabResources;

impl AlbertModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/model",
        "https://huggingface.co/albert-base-v2/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/model",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/rust_model.ot",
    );
}

impl AlbertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/config",
        "https://huggingface.co/albert-base-v2/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/config",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/config.json",
    );
}

impl AlbertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/ALBERT>. Modified with conversion to C-array format.
    pub const ALBERT_BASE_V2: (&'static str, &'static str) = (
        "albert-base-v2/spiece",
        "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/spiece",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/spiece.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # ALBERT model configuration
/// Defines the ALBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct AlbertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub classifier_dropout_prob: Option<f64>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub embedding_size: i64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub pad_token_id: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for AlbertConfig {}

impl Default for AlbertConfig {
    fn default() -> Self {
        AlbertConfig {
            hidden_act: Activation::gelu_new,
            attention_probs_dropout_prob: 0.0,
            classifier_dropout_prob: Some(0.1),
            bos_token_id: 2,
            eos_token_id: 3,
            embedding_size: 128,
            hidden_dropout_prob: 0.0,
            hidden_size: 4096,
            initializer_range: 0.02,
            inner_group_num: 1,
            intermediate_size: 16384,
            layer_norm_eps: Some(1e-12),
            max_position_embeddings: 512,
            num_attention_heads: 64,
            num_hidden_groups: 1,
            num_hidden_layers: 12,
            pad_token_id: 0,
            type_vocab_size: 2,
            vocab_size: 30000,
            output_attentions: None,
            output_hidden_states: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
        }
    }
}
