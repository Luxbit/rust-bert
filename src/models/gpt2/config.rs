//! Configuration and pre-trained resource definitions for the Gpt2 model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};

/// # GPT2 Pretrained model weight files
pub struct Gpt2ModelResources;
/// # GPT2 Pretrained model config files
pub struct Gpt2ConfigResources;
/// # GPT2 Pretrained model vocab files
pub struct Gpt2VocabResources;
/// # GPT2 Pretrained model merges files
pub struct Gpt2MergesResources;
#[derive(Debug, Serialize, Deserialize, Clone)]
/// # GPT2 model configuration
/// Defines the GPT2 model architecture (e.g. number of layers, hidden layer size, vocab size...).
/// Shared between GPT and GPT2 models
pub struct Gpt2Config {
    pub attn_pdrop: Option<f64>,
    pub embd_pdrop: Option<f64>,
    pub hidden_dropout_prob: Option<f64>,
    pub afn: Option<Activation>,
    pub initializer_range: f64,
    pub layer_norm_epsilon: f64,
    pub n_ctx: i64,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_layer: i64,
    pub n_positions: i64,
    pub num_labels: Option<i64>,
    pub output_past: Option<bool>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub resid_pdrop: Option<f64>,
    pub vocab_size: i64,
    pub decoder_start_token_id: Option<i64>,
    pub forced_bos_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
}

impl Config for Gpt2Config {}

impl Default for Gpt2Config {
    fn default() -> Self {
        Gpt2Config {
            attn_pdrop: Some(0.1),
            embd_pdrop: Some(0.1),
            hidden_dropout_prob: None,
            afn: Some(Activation::gelu_new),
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            n_ctx: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_positions: 0,
            num_labels: None,
            output_past: None,
            output_attentions: None,
            output_hidden_states: None,
            resid_pdrop: Some(0.1),
            vocab_size: 50257,
            decoder_start_token_id: None,
            forced_bos_token_id: None,
            forced_eos_token_id: None,
        }
    }
}
