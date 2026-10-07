//! Configuration and pre-trained resource definitions for the MobileBert model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # MobileBERT Pretrained model weight files
pub struct MobileBertModelResources;
/// # MobileBERT Pretrained model config files
pub struct MobileBertConfigResources;
/// # MobileBERT Pretrained model vocab files
pub struct MobileBertVocabResources;
#[derive(Debug, Serialize, Deserialize)]
/// # MobileBERT model configuration
/// Defines the MobileBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MobileBertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub embedding_size: i64,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_idx: Option<i64>,
    pub trigram_input: Option<bool>,
    pub use_bottleneck: Option<bool>,
    pub use_bottleneck_attention: Option<bool>,
    pub intra_bottleneck_size: Option<i64>,
    pub key_query_shared_bottleneck: Option<bool>,
    pub num_feedforward_networks: Option<i64>,
    pub normalization_type: Option<NormalizationType>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for MobileBertConfig {}

impl Default for MobileBertConfig {
    fn default() -> Self {
        MobileBertConfig {
            hidden_act: Activation::relu,
            attention_probs_dropout_prob: 0.1,
            hidden_dropout_prob: 0.0,
            hidden_size: 512,
            initializer_range: 0.02,
            intermediate_size: 512,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_hidden_layers: 24,
            type_vocab_size: 2,
            vocab_size: 30522,
            embedding_size: 128,
            layer_norm_eps: Some(1e-12),
            pad_token_idx: Some(0),
            trigram_input: Some(true),
            use_bottleneck: Some(true),
            use_bottleneck_attention: Some(false),
            intra_bottleneck_size: Some(128),
            key_query_shared_bottleneck: Some(true),
            num_feedforward_networks: Some(4),
            normalization_type: Some(NormalizationType::no_norm),
            output_attentions: None,
            output_hidden_states: None,
            classifier_activation: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Normalization type to use for the MobileBERT model.
/// `no_norm` uses a matrix multiplication with a set of learned weights, while `layer_norm` uses a
/// build-in layer normalization module.
pub enum NormalizationType {
    layer_norm,
    no_norm,
}
