// Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};

use crate::pipelines::translation::Language;
use crate::Config;

/// # T5 Pretrained model weight files
pub struct T5ModelResources;

/// # T5 Pretrained model config files
pub struct T5ConfigResources;

/// # T5 Pretrained model vocab files
pub struct T5VocabResources;

/// # T5 optional prefixes
pub struct T5Prefix;

/// # T5 source languages pre-sets
pub struct T5SourceLanguages;

/// # T5 target languages pre-sets
pub type T5TargetLanguages = T5SourceLanguages;

impl T5ModelResources {
    /// Shared under Apache 2.0 license by the T5 Authors at <https://github.com/google-research/text-to-text-transfer-transformer>. Modified with conversion to C-array format.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/model",
        "https://huggingface.co/t5-small/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the T5 Authors at <https://github.com/google-research/text-to-text-transfer-transformer>. Modified with conversion to C-array format.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/model",
        "https://huggingface.co/t5-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/model",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/rust_model.ot",
    );
}

impl T5ConfigResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/config",
        "https://huggingface.co/t5-small/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/config",
        "https://huggingface.co/t5-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/config.json",
    );
}

impl T5VocabResources {
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_SMALL: (&'static str, &'static str) = (
        "t5-small/spiece",
        "https://huggingface.co/t5-small/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license by the Google team at <https://github.com/google-research/text-to-text-transfer-transformer>.
    pub const T5_BASE: (&'static str, &'static str) = (
        "t5-base/spiece",
        "https://huggingface.co/t5-base/resolve/main/spiece.model",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/spiece",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/spiece.model",
    );
}

const T5LANGUAGES: [Language; 3] = [Language::English, Language::French, Language::German];

impl T5SourceLanguages {
    pub const T5_SMALL: [Language; 3] = T5LANGUAGES;
    pub const T5_BASE: [Language; 3] = T5LANGUAGES;
}

impl T5Prefix {
    pub const ENGLISH2FRENCH: Option<&'static str> = Some("translate English to French:");
    pub const ENGLISH2GERMAN: Option<&'static str> = Some("translate English to German:");
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(rename_all = "kebab-case")]
/// # Options for T5 Feed-forward projection layer
pub enum FeedForwardProj {
    /// ReLU
    Relu,
    /// Gated geLU
    GatedGelu,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # T5 model configuration
/// Defines the T5 model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct T5Config {
    pub dropout_rate: f64,
    pub d_model: i64,
    pub d_ff: i64,
    pub d_kv: i64,
    pub decoder_start_token_id: Option<i64>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub forced_bos_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
    pub initializer_factor: f64,
    pub is_encoder_decoder: Option<bool>,
    pub layer_norm_epsilon: f64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub output_past: Option<bool>,
    pub pad_token_id: Option<i64>,
    pub relative_attention_num_buckets: i64,
    pub relative_attention_max_distance: Option<i64>,
    pub vocab_size: i64,
    pub feed_forward_proj: Option<FeedForwardProj>,
    pub tie_word_embeddings: Option<bool>,
    pub task_specific_params: Option<TaskSpecificParams>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

/// # T5 task-specific configurations
/// Defines the T5 configuration for summarization and translation tasks
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskSpecificParams {
    summarization: Summarization,
    translation_en_to_de: TranslationEnToDe,
    translation_en_to_fr: TranslationEnToFr,
    translation_en_to_ro: TranslationEnToRo,
}

/// # T5 summarization configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Summarization {
    early_stopping: bool,
    length_penalty: f64,
    max_length: i64,
    min_length: i64,
    no_repeat_ngram_size: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to German configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToDe {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to French configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToFr {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

/// # T5 English to Romanian configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranslationEnToRo {
    early_stopping: bool,
    max_length: i64,
    num_beams: i64,
    prefix: String,
}

impl Config for T5Config {}

impl Default for T5Config {
    fn default() -> Self {
        T5Config {
            dropout_rate: 0.1,
            d_model: 512,
            d_ff: 2048,
            d_kv: 64,
            decoder_start_token_id: None,
            bos_token_id: None,
            eos_token_id: Some(1),
            forced_bos_token_id: None,
            forced_eos_token_id: None,
            initializer_factor: 1.0,
            is_encoder_decoder: None,
            layer_norm_epsilon: 1e-6,
            num_heads: 8,
            num_layers: 6,
            output_past: None,
            pad_token_id: Some(0),
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: Some(128),
            vocab_size: 32128,
            feed_forward_proj: Some(FeedForwardProj::Relu),
            tie_word_embeddings: None,
            task_specific_params: None,
            output_attentions: None,
            output_hidden_states: None,
        }
    }
}
