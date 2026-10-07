//! Configuration and pre-trained resource definitions for the DistilBert model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # DistilBERT Pretrained model weight files
pub struct DistilBertModelResources;

/// # DistilBERT Pretrained model config files
pub struct DistilBertConfigResources;

/// # DistilBERT Pretrained model vocab files
pub struct DistilBertVocabResources;

impl DistilBertModelResources {
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = (
        "distilbert-sst2/model",
        "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT: (&'static str, &'static str) = (
        "distilbert/model",
        "https://huggingface.co/distilbert-base-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = (
        "distilbert-qa/model",
        "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/model",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/rust_model.ot",
    );
}

impl DistilBertConfigResources {
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = (
        "distilbert-sst2/config",
        "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT: (&'static str, &'static str) = (
        "distilbert/config",
        "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = (
        "distilbert-qa/config",
        "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/config.json",
    );
}

impl DistilBertVocabResources {
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SST2: (&'static str, &'static str) = (
        "distilbert-sst2/vocab",
        "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT: (&'static str, &'static str) = (
        "distilbert/vocab",
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by the HuggingFace Inc. team at <https://huggingface.co/models>. Modified with conversion to C-array format.
    pub const DISTIL_BERT_SQUAD: (&'static str, &'static str) = (
        "distilbert-qa/vocab",
        "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/vocab",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/vocab.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # DistilBERT model configuration
/// Defines the DistilBERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DistilBertConfig {
    pub activation: Activation,
    pub attention_dropout: f64,
    pub dim: i64,
    pub dropout: f64,
    pub hidden_dim: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub initializer_range: f32,
    pub is_decoder: Option<bool>,
    pub label2id: Option<HashMap<String, i64>>,
    pub max_position_embeddings: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub qa_dropout: f64,
    pub seq_classif_dropout: f64,
    pub sinusoidal_pos_embds: bool,
    pub tie_weights_: bool,
    pub vocab_size: i64,
}

impl Config for DistilBertConfig {}

impl Default for DistilBertConfig {
    fn default() -> Self {
        DistilBertConfig {
            activation: Activation::gelu,
            attention_dropout: 0.1,
            dim: 768,
            dropout: 0.1,
            hidden_dim: 3072,
            id2label: None,
            initializer_range: 0.02,
            is_decoder: None,
            label2id: None,
            max_position_embeddings: 512,
            n_heads: 12,
            n_layers: 6,
            output_attentions: None,
            output_hidden_states: None,
            output_past: None,
            qa_dropout: 0.1,
            seq_classif_dropout: 0.2,
            sinusoidal_pos_embds: false,
            tie_weights_: false,
            vocab_size: 30522,
        }
    }
}
