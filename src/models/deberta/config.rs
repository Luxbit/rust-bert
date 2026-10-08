//! Configuration and pre-trained resource definitions for the Deberta model.
use crate::common::activations::Activation;
use crate::common::error::RustBertError;
use crate::Config;
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// # DeBERTa Pretrained model weight files
pub struct DebertaModelResources;
/// # DeBERTa Pretrained model config files
pub struct DebertaConfigResources;
/// # DeBERTa Pretrained model vocab files
pub struct DebertaVocabResources;
/// # DeBERTa Pretrained model merges files
pub struct DebertaMergesResources;
impl Config for DebertaConfig {}

#[derive(Debug, Serialize, Deserialize)]
/// # DeBERTa model configuration
/// Defines the DeBERTa model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct DebertaConfig {
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
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
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
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub share_att_key: Option<bool>,
    pub position_buckets: Option<i64>,
}

impl Default for DebertaConfig {
    fn default() -> Self {
        DebertaConfig {
            hidden_act: Activation::gelu,
            attention_probs_dropout_prob: 0.1,
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            type_vocab_size: 0,
            vocab_size: 50265,
            position_biased_input: Some(true),
            pos_att_type: None,
            pooler_dropout: Some(0.0),
            pooler_hidden_act: Some(Activation::gelu),
            pooler_hidden_size: Some(768),
            layer_norm_eps: Some(1e-7),
            pad_token_id: Some(0),
            relative_attention: Some(false),
            max_relative_positions: Some(-1),
            embedding_size: None,
            talking_head: None,
            output_hidden_states: None,
            output_attentions: None,
            classifier_dropout: None,
            is_decoder: None,
            id2label: None,
            label2id: None,
            share_att_key: None,
            position_buckets: None,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
/// # Position attention type to use for the DeBERTa model.
pub enum PositionAttentionType {
    p2c,
    c2p,
    p2p,
}

impl FromStr for PositionAttentionType {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "p2c" => Ok(PositionAttentionType::p2c),
            "c2p" => Ok(PositionAttentionType::c2p),
            "p2p" => Ok(PositionAttentionType::p2p),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Position attention type `{s}` not in accepted variants (`p2c`, `c2p`, `p2p`)",
            ))),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PositionAttentionTypes {
    types: Vec<PositionAttentionType>,
}

impl FromStr for PositionAttentionTypes {
    type Err = RustBertError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(PositionAttentionType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PositionAttentionTypes { types })
    }
}

impl PositionAttentionTypes {
    pub fn has_type(&self, attention_type: PositionAttentionType) -> bool {
        self.types.contains(&attention_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

pub fn deserialize_attention_type<'de, D>(
    deserializer: D,
) -> Result<Option<PositionAttentionTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct AttentionTypeVisitor;

    impl<'de> Visitor<'de> for AttentionTypeVisitor {
        type Value = PositionAttentionTypes;

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
            while let Some(attention_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(attention_type.as_str()).unwrap())
            }
            Ok(PositionAttentionTypes { types })
        }
    }

    deserializer.deserialize_any(AttentionTypeVisitor).map(Some)
}
