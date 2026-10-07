//! Configuration and pre-trained resource definitions for the DebertaV2 model.
use crate::common::activations::Activation;
use crate::common::error::RustBertError;
use crate::deberta::config::{deserialize_attention_type, PositionAttentionTypes};
use crate::Config;
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// # DeBERTaV2 Pretrained model weight files
pub struct DebertaV2ModelResources;

/// # DeBERTaV2 Pretrained model config files
pub struct DebertaV2ConfigResources;

/// # DeBERTaV2 Pretrained model vocab files
pub struct DebertaV2VocabResources;

impl DebertaV2ModelResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/model",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/rust_model.ot",
    );
}

impl DebertaV2ConfigResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/config",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/config.json",
    );
}

impl DebertaV2VocabResources {
    /// Shared under MIT license by the Microsoft team at <https://huggingface.co/microsoft/deberta-v3-base>. Modified with conversion to C-array format.
    pub const DEBERTA_V3_BASE: (&'static str, &'static str) = (
        "deberta-v3-base/vocab",
        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model",
    );
}

impl Config for DebertaV2Config {}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa (v2) model configuration
/// Defines the DeBERTa (v2) model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaV2Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub position_buckets: Option<i64>,
    pub num_attention_heads: i64,
    pub type_vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    #[serde(default, deserialize_with = "deserialize_norm_type")]
    pub norm_rel_ebd: Option<NormRelEmbedTypes>,
    pub share_att_key: Option<bool>,
    pub conv_kernel_size: Option<i64>,
    pub conv_groups: Option<i64>,
    pub conv_act: Option<Activation>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub embedding_size: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
/// # Layer normalization layer for the DeBERTa model's relative embeddings.
pub enum NormRelEmbedType {
    layer_norm,
}

impl FromStr for NormRelEmbedType {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "layer_norm" => Ok(NormRelEmbedType::layer_norm),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Layer normalization type `{s}` not in accepted variants (`layer_norm`)",
            ))),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NormRelEmbedTypes {
    types: Vec<NormRelEmbedType>,
}

impl FromStr for NormRelEmbedTypes {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(NormRelEmbedType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NormRelEmbedTypes { types })
    }
}

impl NormRelEmbedTypes {
    pub fn has_type(&self, norm_type: NormRelEmbedType) -> bool {
        self.types.contains(&norm_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }
}

pub fn deserialize_norm_type<'de, D>(deserializer: D) -> Result<Option<NormRelEmbedTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct NormTypeVisitor;

    impl<'de> Visitor<'de> for NormTypeVisitor {
        type Value = NormRelEmbedTypes;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, string or sequence")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).unwrap())
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut types = vec![];
            while let Some(norm_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(norm_type.as_str()).unwrap())
            }
            Ok(NormRelEmbedTypes { types })
        }
    }

    deserializer.deserialize_any(NormTypeVisitor).map(Some)
}
