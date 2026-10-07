//! Configuration and pre-trained resource definitions for the Bert model.
use crate::common::activations::Activation;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # BERT Pretrained model weight files
pub struct BertModelResources;

/// # BERT Pretrained model config files
pub struct BertConfigResources;

/// # BERT Pretrained model vocab files
pub struct BertVocabResources;

impl BertModelResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/model",
        "https://huggingface.co/bert-base-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT_LARGE: (&'static str, &'static str) = (
        "bert-large/model",
        "https://huggingface.co/bert-large-uncased/resolve/main/rust_model.ot",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/model",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/model",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/model",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/model",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/model",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/rust_model.ot",
    );
}

impl BertConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/config",
        "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT_LARGE: (&'static str, &'static str) = (
        "bert-large/config",
        "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/config",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/config",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
    );
}

impl BertVocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT: (&'static str, &'static str) = (
        "bert/vocab",
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/bert>. Modified with conversion to C-array format.
    pub const BERT_LARGE: (&'static str, &'static str) = (
        "bert-large/vocab",
        "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
    );
    /// Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at <https://github.com/dbmdz/berts>. Modified with conversion to C-array format.
    pub const BERT_NER: (&'static str, &'static str) = (
        "bert-ner/vocab",
        "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license by Hugging Face Inc at <https://github.com/huggingface/transformers/tree/master/examples/question-answering>. Modified with conversion to C-array format.
    pub const BERT_QA: (&'static str, &'static str) = (
        "bert-qa/vocab",
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/vocab",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/vocab",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/vocab.txt",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/vocab",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # BERT model configuration
/// Defines the BERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct BertConfig {
    pub hidden_act: Activation,
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
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

impl Config for BertConfig {}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            hidden_act: Activation::gelu,
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
            output_attentions: None,
            output_hidden_states: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
        }
    }
}
