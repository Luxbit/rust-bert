// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Question Answering pipeline
//! Extractive question answering from a given question and context. By default, the dependencies for this
//! model will be downloaded for a DistilBERT model finetuned on SQuAD (Stanford Question Answering Dataset).
//! Customized DistilBERT models can be loaded by overwriting the resources in the configuration.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/distilbert-qa
//!
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//!
//! let answers = qa_model.predict(&vec![QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//!
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::question_answering::Answer;
//! # let output =
//! [Answer {
//!     score: 0.9976,
//!     start: 13,
//!     end: 21,
//!     answer: String::from("Amsterdam"),
//! }]
//! # ;
//! ```

use crate::common::error::RustBertError;
#[cfg(feature = "libtorch")]
use crate::pipelines::common::ConfigOption;
use crate::pipelines::common::{ModelResource, ModelType, TokenizerOption};
use crate::resources::ResourceProvider;
use rust_tokenizers::{Offset, TokenIdsWithOffsets, TokenizedInput};
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "libtorch")]
use crate::pipelines::common::cast_var_store;
#[cfg(feature = "onnx")]
use crate::pipelines::onnx::{config::ONNXEnvironmentConfig, ONNXEncoder};
#[cfg(feature = "libtorch")]
use tch::nn::VarStore;
#[cfg(feature = "libtorch")]
use tch::Kind;

#[cfg(feature = "remote")]
use crate::distilbert::{
    DistilBertConfigResources, DistilBertModelResources, DistilBertVocabResources,
};
#[cfg(feature = "remote")]
use crate::resources::RemoteResource;

#[cfg(feature = "libtorch")]
mod torch_models {
    pub use crate::albert::AlbertForQuestionAnswering;
    pub use crate::bert::BertForQuestionAnswering;
    pub use crate::deberta::DebertaForQuestionAnswering;
    pub use crate::deberta_v2::DebertaV2ForQuestionAnswering;
    pub use crate::distilbert::DistilBertForQuestionAnswering;
    pub use crate::fnet::FNetForQuestionAnswering;
    pub use crate::longformer::LongformerForQuestionAnswering;
    pub use crate::mobilebert::MobileBertForQuestionAnswering;
    pub use crate::reformer::ReformerForQuestionAnswering;
    pub use crate::roberta::RobertaForQuestionAnswering;
    pub use crate::xlnet::XLNetForQuestionAnswering;
}

use crate::Device;
use ndarray::{Array1, Array2, ArrayD};

#[derive(Serialize, Deserialize)]
/// # Input for Question Answering
/// Includes a context (containing the answer) and question strings
pub struct QaInput {
    /// Question string
    pub question: String,
    /// Context or query
    pub context: String,
}

#[derive(Debug)]
struct QaFeature {
    pub input_ids: Vec<i64>,
    pub offsets: Vec<Option<Offset>>,
    pub token_type_ids: Vec<i8>,
    pub p_mask: Vec<i8>,
    pub example_index: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// # Output for Question Answering
pub struct Answer {
    /// Confidence score
    pub score: f64,
    /// Start position of answer span
    pub start: usize,
    /// End position of answer span
    pub end: usize,
    /// Answer span
    pub answer: String,
}

impl PartialEq for Answer {
    fn eq(&self, other: &Self) -> bool {
        (self.start == other.start) && (self.end == other.end) && (self.answer == other.answer)
    }
}

fn remove_duplicates<T: PartialEq + Clone>(vector: &mut Vec<T>) -> &mut Vec<T> {
    let mut potential_duplicates = vec![];
    vector.retain(|item| {
        if potential_duplicates.contains(item) {
            false
        } else {
            potential_duplicates.push(item.clone());
            true
        }
    });
    vector
}

/// # Configuration for question answering
/// Contains information regarding the model to load and device to place the model on.
pub struct QuestionAnsweringConfig {
    /// Model weights resource (default: pretrained DistilBERT model on SQuAD)
    pub model_resource: ModelResource,
    /// Config resource (default: pretrained DistilBERT model on SQuAD)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained DistilBERT model on SQuAD)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: None)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Model type
    pub model_type: ModelType,
    /// Flag indicating if the model expects a lower casing of the input
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Maximum sequence length for the combined query and context
    pub max_seq_length: usize,
    /// Stride to apply if the context needs to be broken down due to a large length. Represents the number of overlapping tokens between sliding windows.
    pub doc_stride: usize,
    /// Maximum length for the query
    pub max_query_length: usize,
    /// Maximum length for the answer
    pub max_answer_length: usize,
    /// Model weights precision (LibTorch backend only). If not provided, will default to full precision on CPU, or the loaded weights precision otherwise
    #[cfg(feature = "libtorch")]
    pub kind: Option<Kind>,
}

impl QuestionAnsweringConfig {
    /// Instantiate a new question answering configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config_resource - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab_resource - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges_resource - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    pub fn new<RC, RV>(
        model_type: ModelType,
        model_resource: ModelResource,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> QuestionAnsweringConfig
    where
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        QuestionAnsweringConfig {
            model_type,
            model_resource,
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            max_seq_length: 384,
            doc_stride: 128,
            max_query_length: 64,
            max_answer_length: 15,
            #[cfg(feature = "libtorch")]
            kind: None,
        }
    }

    /// Instantiate a new question answering configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model_resource - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config_resource - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab_resource - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges_resource - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model)
    /// * max_seq_length - Optional maximum sequence token length to limit memory footprint. If the context is too long, it will be processed with sliding windows. Defaults to 384.
    /// * max_query_length - Optional maximum question token length. Defaults to 64.
    /// * doc_stride - Optional stride to apply if a sliding window is required to process the input context. Represents the number of overlapping tokens between sliding windows. This should be lower than the max_seq_length minus max_query_length (otherwise there is a risk for the sliding window not to progress). Defaults to 128.
    /// * max_answer_length - Optional maximum token length for the extracted answer. Defaults to 15.
    pub fn custom_new<RC, RV>(
        model_type: ModelType,
        model_resource: ModelResource,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
        max_seq_length: impl Into<Option<usize>>,
        doc_stride: impl Into<Option<usize>>,
        max_query_length: impl Into<Option<usize>>,
        max_answer_length: impl Into<Option<usize>>,
    ) -> QuestionAnsweringConfig
    where
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        QuestionAnsweringConfig {
            model_type,
            model_resource,
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            max_seq_length: max_seq_length.into().unwrap_or(384),
            doc_stride: doc_stride.into().unwrap_or(128),
            max_query_length: max_query_length.into().unwrap_or(64),
            max_answer_length: max_answer_length.into().unwrap_or(15),
            #[cfg(feature = "libtorch")]
            kind: None,
        }
    }
}

#[cfg(feature = "remote")]
impl Default for QuestionAnsweringConfig {
    fn default() -> QuestionAnsweringConfig {
        QuestionAnsweringConfig {
            model_resource: ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
                DistilBertModelResources::DISTIL_BERT_SQUAD,
            ))),
            config_resource: Box::new(RemoteResource::from_pretrained(
                DistilBertConfigResources::DISTIL_BERT_SQUAD,
            )),
            vocab_resource: Box::new(RemoteResource::from_pretrained(
                DistilBertVocabResources::DISTIL_BERT_SQUAD,
            )),
            merges_resource: None,
            device: Device::cuda_if_available(),
            #[cfg(feature = "libtorch")]
            kind: None,
            model_type: ModelType::DistilBert,
            lower_case: false,
            add_prefix_space: None,
            strip_accents: None,
            max_seq_length: 384,
            doc_stride: 128,
            max_query_length: 64,
            max_answer_length: 15,
        }
    }
}

#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular question answering model, for any of the supported models
pub enum QuestionAnsweringOption {
    /// Bert for Question Answering
    #[cfg(feature = "libtorch")]
    Bert(torch_models::BertForQuestionAnswering),
    /// DeBERTa for Question Answering
    #[cfg(feature = "libtorch")]
    Deberta(torch_models::DebertaForQuestionAnswering),
    /// DeBERTa V2 for Question Answering
    #[cfg(feature = "libtorch")]
    DebertaV2(torch_models::DebertaV2ForQuestionAnswering),
    /// DistilBert for Question Answering
    #[cfg(feature = "libtorch")]
    DistilBert(torch_models::DistilBertForQuestionAnswering),
    /// MobileBert for Question Answering
    #[cfg(feature = "libtorch")]
    MobileBert(torch_models::MobileBertForQuestionAnswering),
    /// Roberta for Question Answering
    #[cfg(feature = "libtorch")]
    Roberta(torch_models::RobertaForQuestionAnswering),
    /// XLMRoberta for Question Answering
    #[cfg(feature = "libtorch")]
    XLMRoberta(torch_models::RobertaForQuestionAnswering),
    /// Albert for Question Answering
    #[cfg(feature = "libtorch")]
    Albert(torch_models::AlbertForQuestionAnswering),
    /// XLNet for Question Answering
    #[cfg(feature = "libtorch")]
    XLNet(torch_models::XLNetForQuestionAnswering),
    /// Reformer for Question Answering
    #[cfg(feature = "libtorch")]
    Reformer(torch_models::ReformerForQuestionAnswering),
    /// Longformer for Question Answering
    #[cfg(feature = "libtorch")]
    Longformer(torch_models::LongformerForQuestionAnswering),
    /// FNet for Question Answering
    #[cfg(feature = "libtorch")]
    FNet(torch_models::FNetForQuestionAnswering),
    /// ONNX model for Question Answering
    #[cfg(feature = "onnx")]
    ONNX(ONNXEncoder),
}

impl QuestionAnsweringOption {
    /// Instantiate a new question answering model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `QuestionAnsweringConfig` - Question answering pipeline configuration. The type of model created will be inferred from the
    ///   `ModelResources` (Torch or ONNX) and `ModelType` (Architecture for Torch models) variants provided and
    pub fn new(config: &QuestionAnsweringConfig) -> Result<Self, RustBertError> {
        match config.model_resource {
            #[cfg(feature = "libtorch")]
            ModelResource::Torch(_) => Self::new_torch(config),
            #[cfg(feature = "onnx")]
            ModelResource::ONNX(_) => Self::new_onnx(config),
            #[cfg(all(feature = "onnx", not(feature = "libtorch")))]
            _ => Err(RustBertError::InvalidConfigurationError(
                "Torch model resources require the `libtorch` feature".to_string(),
            )),
        }
    }

    #[cfg(feature = "libtorch")]
    fn new_torch(config: &QuestionAnsweringConfig) -> Result<Self, RustBertError> {
        let device: tch::Device = config.device.into();
        let weights_path = config.model_resource.get_torch_local_path()?;
        let mut var_store = VarStore::new(device);
        let model_config = &mut ConfigOption::from_file(
            config.model_type,
            config.config_resource.get_local_path()?,
        );
        let model_type = config.model_type;
        let model = match model_type {
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(QuestionAnsweringOption::Bert(
                        torch_models::BertForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = model_config {
                    Ok(QuestionAnsweringOption::Deberta(
                        torch_models::DebertaForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = model_config {
                    Ok(QuestionAnsweringOption::DebertaV2(
                        torch_models::DebertaV2ForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaV2Config for DeBERTa V2!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(ref mut config) = model_config {
                    config.sinusoidal_pos_embds = false;
                    Ok(QuestionAnsweringOption::DistilBert(
                        torch_models::DistilBertForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DistilBertConfig for DistilBert!".to_string(),
                    ))
                }
            }
            ModelType::MobileBert => {
                if let ConfigOption::MobileBert(config) = model_config {
                    Ok(QuestionAnsweringOption::MobileBert(
                        torch_models::MobileBertForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = model_config {
                    Ok(QuestionAnsweringOption::Roberta(
                        torch_models::RobertaForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(QuestionAnsweringOption::XLMRoberta(
                        torch_models::RobertaForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = model_config {
                    Ok(QuestionAnsweringOption::Albert(
                        torch_models::AlbertForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::XLNet => {
                if let ConfigOption::XLNet(config) = model_config {
                    Ok(QuestionAnsweringOption::XLNet(
                        torch_models::XLNetForQuestionAnswering::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a XLNetConfig for XLNet!".to_string(),
                    ))
                }
            }
            ModelType::Reformer => {
                if let ConfigOption::Reformer(config) = model_config {
                    Ok(QuestionAnsweringOption::Reformer(
                        torch_models::ReformerForQuestionAnswering::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a ReformerConfig for Reformer!".to_string(),
                    ))
                }
            }
            ModelType::Longformer => {
                if let ConfigOption::Longformer(config) = model_config {
                    Ok(QuestionAnsweringOption::Longformer(
                        torch_models::LongformerForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            ModelType::FNet => {
                if let ConfigOption::FNet(config) = model_config {
                    Ok(QuestionAnsweringOption::FNet(
                        torch_models::FNetForQuestionAnswering::new(var_store.root(), config),
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a FNetConfig for FNet!".to_string(),
                    ))
                }
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "QuestionAnswering not implemented for {model_type:?}!",
            ))),
        }?;
        var_store.load(weights_path)?;
        cast_var_store(&mut var_store, config.kind, device);
        Ok(model)
    }

    #[cfg(feature = "onnx")]
    pub fn new_onnx(config: &QuestionAnsweringConfig) -> Result<Self, RustBertError> {
        let onnx_config = ONNXEnvironmentConfig::from_device(config.device);
        let encoder_file = config
            .model_resource
            .get_onnx_local_paths()?
            .encoder_path
            .ok_or(RustBertError::InvalidConfigurationError(
                "An encoder file must be provided for question answering ONNX models.".to_string(),
            ))?;

        Ok(Self::ONNX(ONNXEncoder::new(encoder_file, &onnx_config)?))
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            #[cfg(feature = "libtorch")]
            Self::Bert(_) => ModelType::Bert,
            #[cfg(feature = "libtorch")]
            Self::Deberta(_) => ModelType::Deberta,
            #[cfg(feature = "libtorch")]
            Self::DebertaV2(_) => ModelType::DebertaV2,
            #[cfg(feature = "libtorch")]
            Self::Roberta(_) => ModelType::Roberta,
            #[cfg(feature = "libtorch")]
            Self::XLMRoberta(_) => ModelType::XLMRoberta,
            #[cfg(feature = "libtorch")]
            Self::DistilBert(_) => ModelType::DistilBert,
            #[cfg(feature = "libtorch")]
            Self::MobileBert(_) => ModelType::MobileBert,
            #[cfg(feature = "libtorch")]
            Self::Albert(_) => ModelType::Albert,
            #[cfg(feature = "libtorch")]
            Self::XLNet(_) => ModelType::XLNet,
            #[cfg(feature = "libtorch")]
            Self::Reformer(_) => ModelType::Reformer,
            #[cfg(feature = "libtorch")]
            Self::Longformer(_) => ModelType::Longformer,
            #[cfg(feature = "libtorch")]
            Self::FNet(_) => ModelType::FNet,
            #[cfg(feature = "onnx")]
            Self::ONNX(_) => ModelType::ONNX,
        }
    }

    /// Interface method to forward_t() of the particular models.
    pub fn forward_t(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        mask: Option<&ArrayD<i64>>,
        input_embeds: Option<&ArrayD<f32>>,
        _token_type_ids: Option<&ArrayD<i64>>,
        train: bool,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let _ = train;
        #[cfg(feature = "libtorch")]
        {
            use crate::common::tensor_conversion::tensor_to_array_f32;
            use crate::pipelines::common::{to_tensor_f32, to_tensor_i64};
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let input_ids_array = input_ids;
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let mask_array = mask;
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let token_type_ids_array = _token_type_ids;
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let input_embeds_array = input_embeds;
            let input_ids = to_tensor_i64(input_ids);
            let mask = to_tensor_i64(mask);
            let _token_type_ids = to_tensor_i64(_token_type_ids);
            let input_embeds = to_tensor_f32(input_embeds);
            match *self {
                Self::Bert(ref model) => {
                    let outputs = model.forward_t(
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        None,
                        input_embeds.as_ref(),
                        train,
                    );
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::Deberta(ref model) => {
                    let outputs = model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            None,
                            None,
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in Deberta forward_t");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::DebertaV2(ref model) => {
                    let outputs = model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            None,
                            None,
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in Deberta V2 forward_t");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::DistilBert(ref model) => {
                    let outputs = model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in distilbert forward_t");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::MobileBert(ref model) => {
                    let outputs = model
                        .forward_t(
                            input_ids.as_ref(),
                            None,
                            None,
                            input_embeds.as_ref(),
                            mask.as_ref(),
                            train,
                        )
                        .expect("Error in mobilebert forward_t");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                #[cfg(feature = "libtorch")]
                Self::Roberta(ref model) | Self::XLMRoberta(ref model) => {
                    let outputs = model.forward_t(
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        None,
                        input_embeds.as_ref(),
                        train,
                    );
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::Albert(ref model) => {
                    let outputs = model.forward_t(
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        None,
                        input_embeds.as_ref(),
                        train,
                    );
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::XLNet(ref model) => {
                    let outputs = model.forward_t(
                        input_ids.as_ref(),
                        mask.as_ref(),
                        None,
                        None,
                        None,
                        None,
                        input_embeds.as_ref(),
                        train,
                    );
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::Reformer(ref model) => {
                    let outputs = model
                        .forward_t(input_ids.as_ref(), None, None, mask.as_ref(), None, train)
                        .expect("Error in reformer forward pass");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::Longformer(ref model) => {
                    let outputs = model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            None,
                            None,
                            None,
                            None,
                            train,
                        )
                        .expect("Error in reformer forward pass");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                Self::FNet(ref model) => {
                    let outputs = model
                        .forward_t(input_ids.as_ref(), None, None, None, train)
                        .expect("Error in fnet forward pass");
                    (
                        tensor_to_array_f32(&outputs.start_logits)
                            .expect("Error converting model output to ndarray"),
                        tensor_to_array_f32(&outputs.end_logits)
                            .expect("Error converting model output to ndarray"),
                    )
                }
                #[cfg(feature = "onnx")]
                Self::ONNX(ref model) => {
                    let outputs = model
                        .forward(
                            input_ids_array,
                            mask_array,
                            token_type_ids_array,
                            None,
                            input_embeds_array,
                        )
                        .expect("Error in ONNX forward pass.");
                    (outputs.start_logits.unwrap(), outputs.end_logits.unwrap())
                }
            }
        }
        #[cfg(not(feature = "libtorch"))]
        {
            match *self {
                #[cfg(feature = "onnx")]
                Self::ONNX(ref model) => {
                    let outputs = model
                        .forward(input_ids, mask, _token_type_ids, None, input_embeds)
                        .expect("Error in ONNX forward pass.");
                    (outputs.start_logits.unwrap(), outputs.end_logits.unwrap())
                }
                #[cfg(not(feature = "onnx"))]
                _ => unreachable!("no inference backend available"),
            }
        }
    }
}

/// # QuestionAnsweringModel to perform extractive question answering
pub struct QuestionAnsweringModel {
    tokenizer: TokenizerOption,
    pad_idx: i64,
    sep_idx: i64,
    max_seq_len: usize,
    doc_stride: usize,
    max_query_length: usize,
    max_answer_len: usize,
    qa_model: QuestionAnsweringOption,
}

impl QuestionAnsweringModel {
    /// Build a new `QuestionAnsweringModel`
    ///
    /// # Arguments
    ///
    /// * `question_answering_config` - `QuestionAnsweringConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
    ///
    /// let qa_model = QuestionAnsweringModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        question_answering_config: QuestionAnsweringConfig,
    ) -> Result<QuestionAnsweringModel, RustBertError> {
        let vocab_path = question_answering_config.vocab_resource.get_local_path()?;
        let merges_path = question_answering_config
            .merges_resource
            .as_ref()
            .map(|resource| resource.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            question_answering_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            question_answering_config.lower_case,
            question_answering_config.strip_accents,
            question_answering_config.add_prefix_space,
        )?;
        Self::new_with_tokenizer(question_answering_config, tokenizer)
    }

    /// Build a new `QuestionAnsweringModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `question_answering_config` - `QuestionAnsweringConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for question answering.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::question_answering::QuestionAnsweringModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///  ModelType::Bert,
    ///  "path/to/vocab.txt",
    ///  None,
    ///  false,
    ///  None,
    ///  None,
    /// )?;
    /// let qa_model = QuestionAnsweringModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        question_answering_config: QuestionAnsweringConfig,
        tokenizer: TokenizerOption,
    ) -> Result<QuestionAnsweringModel, RustBertError> {
        let qa_model = QuestionAnsweringOption::new(&question_answering_config)?;

        let pad_idx = tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for Question Answering should contain a PAD id");
        let sep_idx = tokenizer
            .get_sep_id()
            .expect("The Tokenizer used for Question Answering should contain a SEP id");

        if question_answering_config.max_seq_length
            < (question_answering_config.max_query_length
                + question_answering_config.doc_stride
                + 24)
        {
            return Err(RustBertError::InvalidConfigurationError(format!(
                "This configuration could cause an excessive number of sliding windows generated.\
                Please ensure max_seq_length > max_query_length + doc_stride + 24.\
                Got max_seq_length: {}, max_query_length: {}, doc_stride: {}",
                question_answering_config.max_seq_length,
                question_answering_config.max_query_length,
                question_answering_config.doc_stride
            )));
        }
        Ok(QuestionAnsweringModel {
            tokenizer,
            pad_idx,
            sep_idx,
            max_seq_len: question_answering_config.max_seq_length,
            doc_stride: question_answering_config.doc_stride,
            max_query_length: question_answering_config.max_query_length,
            max_answer_len: question_answering_config.max_answer_length,
            qa_model,
        })
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }

    /// Perform extractive question answering given a list of `QaInputs`
    ///
    /// # Arguments
    ///
    /// * `qa_inputs` - `&[QaInput]` Array of Question Answering inputs (context and question pairs)
    /// * `top_k` - return the top-k answers for each QaInput. Set to 1 to return only the best answer.
    /// * `batch_size` - maximum batch size for the model forward pass.
    ///
    /// # Returns
    /// * `Vec<Vec<Answer>>` Vector (same length as `qa_inputs`) of vectors (each of length `top_k`) containing the extracted answers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
    ///
    /// let qa_model = QuestionAnsweringModel::new(Default::default())?;
    ///
    /// let question_1 = String::from("Where does Amy live ?");
    /// let context_1 = String::from("Amy lives in Amsterdam");
    /// let question_2 = String::from("Where does Eric live");
    /// let context_2 = String::from("While Amy lives in Amsterdam, Eric is in The Hague.");
    ///
    /// let qa_input_1 = QaInput {
    ///  question: question_1,
    ///  context: context_1,
    /// };
    /// let qa_input_2 = QaInput {
    ///  question: question_2,
    ///  context: context_2,
    /// };
    /// let answers = qa_model.predict(&[qa_input_1, qa_input_2], 1, 32);
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        qa_inputs: &[QaInput],
        top_k: i64,
        batch_size: usize,
    ) -> Vec<Vec<Answer>> {
        let mut features: Vec<QaFeature> = qa_inputs
            .iter()
            .enumerate()
            .flat_map(|(example_index, qa_example)| {
                self.generate_features(
                    qa_example,
                    self.max_seq_len,
                    self.doc_stride,
                    self.max_query_length,
                    example_index as i64,
                )
            })
            .collect();

        let mut example_top_k_answers_map: HashMap<usize, Vec<Answer>> = HashMap::new();
        let mut start = 0usize;
        let len_features = features.len();

        while start < len_features {
            let end = start + min(len_features - start, batch_size);
            let batch_features = &mut features[start..end];
            {
                let (input_ids, attention_masks, token_type_ids) =
                    self.pad_features(batch_features);

                let (start_logits, end_logits) = self.qa_model.forward_t(
                    Some(&input_ids.into_dyn()),
                    Some(&attention_masks.into_dyn()),
                    None,
                    Some(&token_type_ids.into_dyn()),
                    false,
                );

                let example_index_to_feature_end_position: Vec<(usize, i64)> = batch_features
                    .iter()
                    .enumerate()
                    .map(|(feature_index, feature)| {
                        (feature.example_index as usize, feature_index as i64 + 1)
                    })
                    .collect();

                let mut feature_id_start = 0;

                for (example_id, max_feature_id) in example_index_to_feature_end_position {
                    let mut answers: Vec<Answer> = vec![];
                    let example = &qa_inputs[example_id];
                    for feature_idx in feature_id_start..max_feature_id {
                        let feature = &batch_features[feature_idx as usize];
                        let start_row =
                            start_logits.index_axis(ndarray::Axis(0), feature_idx as usize);
                        let end_row = end_logits.index_axis(ndarray::Axis(0), feature_idx as usize);
                        // Mask special tokens / padding positions out of the softmax
                        let start_scores: Vec<f32> = start_row
                            .iter()
                            .zip(feature.p_mask.iter())
                            .map(|(&logit, &mask)| if mask == 1 { f32::MIN } else { logit })
                            .collect();
                        let end_scores: Vec<f32> = end_row
                            .iter()
                            .zip(feature.p_mask.iter())
                            .map(|(&logit, &mask)| if mask == 1 { f32::MIN } else { logit })
                            .collect();
                        let start_probs = crate::common::tensor_ops::softmax_last_dim(
                            &Array1::from(start_scores).into_dyn(),
                        );
                        let end_probs = crate::common::tensor_ops::softmax_last_dim(
                            &Array1::from(end_scores).into_dyn(),
                        );

                        let (starts, ends, scores) = self.decode(
                            start_probs.as_slice().unwrap(),
                            end_probs.as_slice().unwrap(),
                            top_k,
                        );

                        for idx in 0..starts.len() {
                            let start_pos = feature.offsets[starts[idx] as usize]
                                .unwrap_or(Offset { begin: 0, end: 0 })
                                .begin as usize;
                            let end_pos = feature.offsets[ends[idx] as usize]
                                .unwrap_or(Offset { begin: 0, end: 0 })
                                .end as usize;
                            let answer = example
                                .context
                                .chars()
                                .take(end_pos)
                                .skip(start_pos)
                                .collect::<String>();

                            answers.push(Answer {
                                score: scores[idx],
                                start: start_pos,
                                end: end_pos,
                                answer,
                            });
                        }
                    }
                    feature_id_start = max_feature_id;
                    let example_answers = example_top_k_answers_map.entry(example_id).or_default();
                    example_answers.extend(answers);
                }
            }
            start = end;
        }
        let mut all_answers = vec![];
        for example_id in 0..qa_inputs.len() {
            if let Some(answers) = example_top_k_answers_map.get_mut(&example_id) {
                remove_duplicates(answers).sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                all_answers.push(answers[..min(answers.len(), top_k as usize)].to_vec());
            } else {
                all_answers.push(vec![]);
            }
        }
        all_answers
    }

    fn decode(&self, start: &[f32], end: &[f32], top_k: i64) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let start_dim = start.len();
        let end_dim = end.len();
        // Outer product of start and end probabilities, keeping only spans with
        // length <= max_answer_len (equivalent to triu(0).tril(max_answer_len - 1))
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(start_dim * end_dim);
        for (i, &start_value) in start.iter().enumerate() {
            let max_end = min(end_dim, i + self.max_answer_len);
            for (j, &end_value) in end.iter().enumerate().take(max_end).skip(i) {
                candidates.push((start_value * end_value, i * end_dim + j));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let count = min(candidates.len(), top_k.max(0) as usize);
        let mut starts: Vec<i64> = vec![];
        let mut ends: Vec<i64> = vec![];
        let mut scores: Vec<f64> = vec![];
        for &(score, flat_index) in candidates.iter().take(count) {
            scores.push(score as f64);
            starts.push((flat_index / end_dim) as i64);
            ends.push((flat_index % end_dim) as i64);
        }
        (starts, ends, scores)
    }

    fn generate_features(
        &self,
        qa_example: &QaInput,
        max_seq_length: usize,
        doc_stride: usize,
        max_query_length: usize,
        example_index: i64,
    ) -> Vec<QaFeature> {
        let mut encoded_query = self.tokenizer.tokenize_with_offsets(&qa_example.question);
        encoded_query.tokens.truncate(max_query_length);
        encoded_query.offsets.truncate(max_query_length);
        encoded_query.reference_offsets.truncate(max_query_length);
        encoded_query.masks.truncate(max_query_length);
        let encoded_query = TokenIdsWithOffsets {
            ids: self.tokenizer.convert_tokens_to_ids(&encoded_query.tokens),
            offsets: encoded_query.offsets,
            reference_offsets: encoded_query.reference_offsets,
            masks: encoded_query.masks,
        };

        let sequence_added_tokens = self
            .tokenizer
            .build_input_with_special_tokens(
                TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                },
                None,
            )
            .token_ids
            .len();

        let sequence_pair_added_tokens = self
            .tokenizer
            .build_input_with_special_tokens(
                TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                },
                Some(TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                }),
            )
            .token_ids
            .len();

        let mut spans: Vec<QaFeature> = vec![];

        let tokenized_context = self.tokenizer.tokenize_with_offsets(&qa_example.context);
        let encoded_context = TokenIdsWithOffsets {
            ids: self
                .tokenizer
                .convert_tokens_to_ids(&tokenized_context.tokens),
            offsets: tokenized_context.offsets,
            reference_offsets: tokenized_context.reference_offsets,
            masks: tokenized_context.masks,
        };
        let max_context_length =
            max_seq_length - sequence_pair_added_tokens - encoded_query.ids.len();

        let mut start_token = 0_usize;
        while (spans.len() * doc_stride) < encoded_context.ids.len() {
            let end_token = min(start_token + max_context_length, encoded_context.ids.len());
            let sub_encoded_context = TokenIdsWithOffsets {
                ids: encoded_context.ids[start_token..end_token].to_vec(),
                offsets: encoded_context.offsets[start_token..end_token].to_vec(),
                reference_offsets: encoded_context.reference_offsets[start_token..end_token]
                    .to_vec(),
                masks: encoded_context.masks[start_token..end_token].to_vec(),
            };

            let encoded_span = self
                .tokenizer
                .build_input_with_special_tokens(encoded_query.clone(), Some(sub_encoded_context));
            let p_mask = self.get_mask(
                &encoded_span,
                encoded_query.ids.len() + sequence_added_tokens,
            );
            let qa_feature = QaFeature {
                input_ids: encoded_span.token_ids,
                offsets: encoded_span.token_offsets,
                token_type_ids: encoded_span.segment_ids,
                p_mask,
                example_index,
            };
            spans.push(qa_feature);
            if end_token == encoded_context.ids.len() {
                break;
            }
            start_token = end_token - doc_stride;
        }
        spans
    }

    fn pad_features(&self, features: &mut [QaFeature]) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
        let max_len = features
            .iter()
            .map(|feature| feature.input_ids.len())
            .max()
            .unwrap();

        let attention_masks = features
            .iter()
            .map(|feature| &feature.input_ids)
            .map(|input| {
                let mut attention_mask = Vec::with_capacity(max_len);
                attention_mask.resize(input.len(), 1);
                attention_mask.resize(max_len, 0);
                attention_mask
            })
            .collect::<Vec<_>>();

        for feature in features.iter_mut() {
            feature.offsets.resize(max_len, None);
            feature.p_mask.resize(max_len, 1);
            feature.input_ids.resize(max_len, self.pad_idx);
            feature
                .token_type_ids
                .resize(max_len, *feature.token_type_ids.last().unwrap_or(&0));
        }

        let mut input_ids = Array2::<i64>::zeros((features.len(), max_len));
        let mut attention_mask_array = Array2::<i64>::zeros((features.len(), max_len));
        let mut token_type_ids = Array2::<i64>::zeros((features.len(), max_len));
        for (row, feature) in features.iter().enumerate() {
            for (col, &id) in feature.input_ids.iter().enumerate() {
                input_ids[[row, col]] = id;
            }
            for (col, &mask) in attention_masks[row].iter().enumerate() {
                attention_mask_array[[row, col]] = mask;
            }
            for (col, &segment) in feature.token_type_ids.iter().enumerate() {
                token_type_ids[[row, col]] = segment as i64;
            }
        }
        (input_ids, attention_mask_array, token_type_ids)
    }

    fn get_mask(&self, encoded_span: &TokenizedInput, question_length: usize) -> Vec<i8> {
        let sep_indices: Vec<usize> = encoded_span
            .token_ids
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == self.sep_idx)
            .map(|(position, _)| position)
            .collect();

        let mut p_mask: Vec<i8> = Vec::with_capacity(encoded_span.token_ids.len());
        p_mask.extend(vec![1; question_length]);
        p_mask.extend(vec![0; encoded_span.token_ids.len() - question_length]);
        for sep_position in sep_indices {
            p_mask[sep_position] = 1;
        }
        p_mask
    }
}

pub fn squad_processor(file_path: PathBuf) -> Vec<QaInput> {
    let file = fs::File::open(file_path).expect("unable to open file");
    let json: serde_json::Value =
        serde_json::from_reader(file).expect("JSON not properly formatted");
    let data = json
        .get("data")
        .expect("SQuAD file does not contain data field")
        .as_array()
        .expect("Data array not properly formatted");

    let mut qa_inputs: Vec<QaInput> = Vec::with_capacity(data.len());
    for qa_input in data.iter() {
        let qa_input = qa_input.as_object().unwrap();
        let paragraphs = qa_input.get("paragraphs").unwrap().as_array().unwrap();
        for paragraph in paragraphs.iter() {
            let paragraph = paragraph.as_object().unwrap();
            let context = paragraph.get("context").unwrap().as_str().unwrap();
            let qas = paragraph.get("qas").unwrap().as_array().unwrap();
            for qa in qas.iter() {
                let question = qa
                    .as_object()
                    .unwrap()
                    .get("question")
                    .unwrap()
                    .as_str()
                    .unwrap();
                qa_inputs.push(QaInput {
                    question: question.to_owned(),
                    context: context.to_owned(),
                });
            }
        }
    }
    qa_inputs
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = QuestionAnsweringConfig::default();
        let _: Box<dyn Send> = Box::new(QuestionAnsweringModel::new(config));
    }
}
