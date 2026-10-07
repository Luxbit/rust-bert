//! Configuration and pre-trained resource definitions for the XLNet model.
use crate::common::activations::Activation;
use crate::common::summary::SummaryType;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # XLNet Pretrained model weight files
pub struct XLNetModelResources;

/// # XLNet Pretrained model config files
pub struct XLNetConfigResources;

/// # XLNet Pretrained model vocab files
pub struct XLNetVocabResources;

impl XLNetModelResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/model",
        "https://huggingface.co/xlnet-base-cased/resolve/main/rust_model.ot",
    );
}

impl XLNetConfigResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/config",
        "https://huggingface.co/xlnet-base-cased/resolve/main/config.json",
    );
}

impl XLNetVocabResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at <https://github.com/zihangdai/xlnet>. Modified with conversion to C-array format.
    pub const XLNET_BASE_CASED: (&'static str, &'static str) = (
        "xlnet-base-cased/spiece",
        "https://huggingface.co/xlnet-base-cased/resolve/main/spiece.model",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Attention type for the model (bidirectional or unidirectional)
pub enum AttentionType {
    /// Bidirectional (XLNet)
    bi,
    /// Unidirectional (Transformer-XL)
    uni,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # XLNet model configuration
/// Defines the XLNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct XLNetConfig {
    pub vocab_size: i64,
    pub d_model: i64,
    pub n_layer: i64,
    pub d_head: i64,
    pub n_head: i64,
    pub d_inner: i64,
    pub ff_activation: Activation,
    pub untie_r: bool,
    pub attn_type: AttentionType,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub dropout: f64,
    pub mem_len: Option<i64>,
    pub reuse_len: Option<i64>,
    pub clamp_len: Option<i64>,
    pub bi_data: bool,
    pub same_length: bool,
    pub summary_type: Option<SummaryType>,
    pub summary_use_proj: Option<bool>,
    pub summary_activation: Option<Activation>,
    pub summary_proj_to_labels: Option<bool>,
    pub summary_first_dropout: Option<f64>,
    pub summary_last_dropout: Option<f64>,
    pub start_n_top: Option<i64>,
    pub end_n_top: Option<i64>,
    pub use_cache: Option<bool>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub chunk_size_feed_forward: Option<i64>,
}

impl Config for XLNetConfig {}

impl Default for XLNetConfig {
    fn default() -> Self {
        XLNetConfig {
            vocab_size: 32000,
            d_model: 1024,
            n_layer: 24,
            d_head: 64,
            n_head: 16,
            d_inner: 4096,
            ff_activation: Activation::gelu,
            untie_r: true,
            attn_type: AttentionType::bi,
            initializer_range: 0.02,
            layer_norm_eps: Some(1e-12),
            dropout: 0.1,
            mem_len: Some(512),
            reuse_len: None,
            clamp_len: Some(-1),
            bi_data: false,
            same_length: false,
            summary_type: Some(SummaryType::last),
            summary_use_proj: Some(true),
            summary_activation: Some(Activation::tanh),
            summary_proj_to_labels: Some(true),
            summary_first_dropout: Some(0.1),
            summary_last_dropout: Some(0.1),
            start_n_top: Some(5),
            end_n_top: Some(5),
            use_cache: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 5,
            id2label: None,
            label2id: None,
            output_attentions: None,
            output_hidden_states: None,
            chunk_size_feed_forward: None,
        }
    }
}
