//! Configuration and pre-trained resource definitions for the Longformer model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # Longformer Pretrained model weight files
pub struct LongformerModelResources;

/// # Longformer Pretrained model config files
pub struct LongformerConfigResources;

/// # Longformer Pretrained model vocab files
pub struct LongformerVocabResources;

/// # Longformer Pretrained model merges files
pub struct LongformerMergesResources;

impl LongformerModelResources {
    /// Shared under Apache 2.0 license by the AllenAI team at <https://github.com/allenai/longformer>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/model",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license at <https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_SQUAD1: (&'static str, &'static str) = (
        "longformer-base-4096/model",
        "https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1/resolve/main/rust_model.ot",
    );
}

impl LongformerConfigResources {
    /// Shared under Apache 2.0 license by the AllenAI team at <https://github.com/allenai/longformer>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/config",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    );
    /// Shared under MIT license at <https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_SQUAD1: (&'static str, &'static str) = (
        "longformer-base-4096/config",
        "https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1/resolve/main/config.json",
    );
}

impl LongformerVocabResources {
    /// Shared under Apache 2.0 license by the AllenAI team at <https://github.com/allenai/longformer>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/vocab",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
    );
    /// Shared under MIT license at <https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_SQUAD1: (&'static str, &'static str) = (
        "longformer-base-4096/vocab",
        "https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1/resolve/main/vocab.json",
    );
}

impl LongformerMergesResources {
    /// Shared under Apache 2.0 license by the AllenAI team at <https://github.com/allenai/longformer>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_4096: (&'static str, &'static str) = (
        "longformer-base-4096/merges",
        "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
    );
    /// Shared under MIT license at <https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1>. Modified with conversion to C-array format.
    pub const LONGFORMER_BASE_SQUAD1: (&'static str, &'static str) = (
        "longformer-base-4096/merges",
        "https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1/resolve/main/merges.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
/// # Longformer Position embeddings type
pub enum PositionEmbeddingType {
    Absolute,
    RelativeKey,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Longformer model configuration
/// Defines the Longformer model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct LongformerConfig {
    pub hidden_act: Activation,
    pub attention_window: Vec<i64>,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub sep_token_id: i64,
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub position_embedding_type: Option<PositionEmbeddingType>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for LongformerConfig {}

impl Default for LongformerConfig {
    fn default() -> Self {
        LongformerConfig {
            hidden_act: Activation::gelu,
            attention_window: vec![512],
            attention_probs_dropout_prob: 0.1,
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30522,
            sep_token_id: 2,
            pad_token_id: None,
            layer_norm_eps: None,
            output_attentions: None,
            output_hidden_states: None,
            position_embedding_type: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
        }
    }
}
