//! Configuration and pre-trained resource definitions for the MBart model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # MBART Pretrained model weight files
pub struct MBartModelResources;
/// # MBART Pretrained model config files
pub struct MBartConfigResources;
/// # MBART Pretrained model vocab files
pub struct MBartVocabResources;
#[derive(Debug, Serialize, Deserialize, Clone)]
/// # MBART model configuration
/// Defines the MBART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MBartConfig {
    pub vocab_size: i64,
    pub max_position_embeddings: i64,
    pub encoder_layers: i64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub decoder_layers: i64,
    pub decoder_ffn_dim: i64,
    pub decoder_attention_heads: i64,
    pub decoder_layerdrop: f64,
    pub is_encoder_decoder: Option<bool>,
    pub activation_function: Option<Activation>,
    pub d_model: i64,
    pub dropout: f64,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classifier_dropout: Option<f64>,
    pub scale_embedding: Option<bool>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub forced_bos_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
    pub decoder_start_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub init_std: f64,
    pub min_length: Option<i64>,
    pub no_repeat_ngram_size: Option<i64>,
    pub normalize_embedding: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
}

impl Config for MBartConfig {}

impl Default for MBartConfig {
    fn default() -> Self {
        MBartConfig {
            vocab_size: 50265,
            max_position_embeddings: 1024,
            encoder_layers: 12,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            encoder_layerdrop: 0.0,
            decoder_layers: 12,
            decoder_ffn_dim: 4096,
            decoder_attention_heads: 16,
            decoder_layerdrop: 0.0,
            is_encoder_decoder: Some(true),
            activation_function: Some(Activation::gelu),
            d_model: 1024,
            dropout: 0.1,
            activation_dropout: 0.0,
            attention_dropout: 0.0,
            classifier_dropout: None,
            scale_embedding: Some(false),
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            pad_token_id: Some(1),
            forced_bos_token_id: None,
            forced_eos_token_id: Some(2),
            decoder_start_token_id: None,
            id2label: None,
            label2id: None,
            init_std: 0.02,
            min_length: None,
            no_repeat_ngram_size: None,
            normalize_embedding: None,
            output_attentions: None,
            output_hidden_states: None,
            output_past: None,
        }
    }
}
