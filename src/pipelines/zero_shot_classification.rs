//! Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
//! Copyright 2019-2020 Guillaume Becquin
//! Copyright 2020 Maarten van Gompel
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!     http://www.apache.org/licenses/LICENSE-2.0
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

//! # Zero-shot classification pipeline
//! Performs zero-shot classification on input sentences with provided labels using a model fine-tuned for Natural Language Inference.
//! The default model is a BART model fine-tuned on a MNLI. From a list of input sequences to classify and a list of target labels,
//! single-class or multi-label classification is performed, translating the classification task to an inference task.
//! The default template for translation to inference task is `This example is about {}.`. This template can be updated to a more specific
//! value that may match better the use case, for example `This review is about a {product_class}`.
//!
//! - `predict` performs single-class classification (one and exactly one label must be true for each provided input)
//! - `predict_multilabel` performs multi-label classification (zero, one or more labels may be true for each provided input)
//!
//! ```no_run
//! # use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
//! # fn main() -> anyhow::Result<()> {
//! let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
//! let input_sentence = "Who are you voting for in 2020?";
//! let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
//! let candidate_labels = &["politics", "public health", "economics", "sports"];
//! let output = sequence_classification_model.predict_multilabel(
//!     &[input_sentence, input_sequence_2],
//!     candidate_labels,
//!     None,
//!     128,
//! );
//! # Ok(())
//! # }
//! ```
//!
//! outputs:
//! ```no_run
//! # use rust_bert::pipelines::sequence_classification::Label;
//! let output = [
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.972,
//!             id: 0,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.032,
//!             id: 1,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "economy".to_string(),
//!             score: 0.006,
//!             id: 2,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.004,
//!             id: 3,
//!             sentence: 0,
//!         },
//!     ],
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.943,
//!             id: 0,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "economy".to_string(),
//!             score: 0.985,
//!             id: 2,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.0818,
//!             id: 1,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.001,
//!             id: 3,
//!             sentence: 1,
//!         },
//!     ],
//! ]
//! .to_vec();
//! ```

use crate::common::tensor_ops::{argmax_last_dim, softmax_last_dim};
#[cfg(feature = "libtorch")]
use crate::pipelines::common::ConfigOption;
use crate::pipelines::common::{ModelResource, ModelType, TokenizerOption};
use crate::pipelines::sequence_classification::Label;
use crate::resources::ResourceProvider;
use crate::Device;
use crate::RustBertError;
use ndarray::{Array1, Array2, ArrayD};
use rust_tokenizers::tokenizer::TruncationStrategy;
use rust_tokenizers::TokenizedInput;

#[cfg(feature = "libtorch")]
use crate::pipelines::common::cast_var_store;
#[cfg(feature = "onnx")]
use crate::pipelines::onnx::{config::ONNXEnvironmentConfig, ONNXEncoder};
#[cfg(all(feature = "remote", feature = "libtorch"))]
use crate::{
    bart::{BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources},
    resources::RemoteResource,
};
#[cfg(feature = "libtorch")]
use tch::nn::VarStore;
#[cfg(feature = "libtorch")]
use tch::Kind;

#[cfg(feature = "libtorch")]
mod torch_models {
    pub use crate::albert::AlbertForSequenceClassification;
    pub use crate::bart::BartForSequenceClassification;
    pub use crate::bert::BertForSequenceClassification;
    pub use crate::deberta::DebertaForSequenceClassification;
    pub use crate::deberta_v2::DebertaV2ForSequenceClassification;
    pub use crate::distilbert::DistilBertModelClassifier;
    pub use crate::longformer::LongformerForSequenceClassification;
    pub use crate::mobilebert::MobileBertForSequenceClassification;
    pub use crate::roberta::RobertaForSequenceClassification;
    pub use crate::xlnet::XLNetForSequenceClassification;
}

/// # Configuration for ZeroShotClassificationModel
/// Contains information regarding the model to load and device to place the model on.
pub struct ZeroShotClassificationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: ModelResource,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: None)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Automatically lower case all input upon tokenization (assumes a lower-cased model)
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Model weights precision (LibTorch backend only). If not provided, will default to full precision on CPU, or the loaded weights precision otherwise
    #[cfg(feature = "libtorch")]
    pub kind: Option<Kind>,
}

impl ZeroShotClassificationConfig {
    /// Instantiate a new zero shot classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `ResourceProvider` pointing to the model to load (e.g.  model.ot)
    /// * config - The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * merges - An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
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
    ) -> ZeroShotClassificationConfig
    where
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
    {
        ZeroShotClassificationConfig {
            model_type,
            model_resource,
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
            #[cfg(feature = "libtorch")]
            kind: None,
        }
    }
}

#[cfg(all(feature = "remote", feature = "libtorch"))]
impl Default for ZeroShotClassificationConfig {
    /// Provides a default zero-shot classification model (English)
    fn default() -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type: ModelType::Bart,
            model_resource: ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
                BartModelResources::BART_MNLI,
            ))),
            config_resource: Box::new(RemoteResource::from_pretrained(
                BartConfigResources::BART_MNLI,
            )),
            vocab_resource: Box::new(RemoteResource::from_pretrained(
                BartVocabResources::BART_MNLI,
            )),
            merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                BartMergesResources::BART_MNLI,
            ))),
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: Device::cuda_if_available(),
            #[cfg(feature = "libtorch")]
            kind: None,
        }
    }
}

/// # Abstraction that holds one particular zero shot classification model, for any of the supported models
/// The models are using a classification architecture that should be trained on Natural Language Inference.
/// The models should output a Tensor of size > 2 in the label dimension, with the first logit corresponding
/// to contradiction and the last logit corresponding to entailment.
#[allow(clippy::large_enum_variant)]
pub enum ZeroShotClassificationOption {
    /// Bart for Sequence Classification
    #[cfg(feature = "libtorch")]
    Bart(torch_models::BartForSequenceClassification),
    /// DeBERTa for Sequence Classification
    #[cfg(feature = "libtorch")]
    Deberta(torch_models::DebertaForSequenceClassification),
    /// DeBERTaV2 for Sequence Classification
    #[cfg(feature = "libtorch")]
    DebertaV2(torch_models::DebertaV2ForSequenceClassification),
    /// Bert for Sequence Classification
    #[cfg(feature = "libtorch")]
    Bert(torch_models::BertForSequenceClassification),
    /// DistilBert for Sequence Classification
    #[cfg(feature = "libtorch")]
    DistilBert(torch_models::DistilBertModelClassifier),
    /// MobileBert for Sequence Classification
    #[cfg(feature = "libtorch")]
    MobileBert(torch_models::MobileBertForSequenceClassification),
    /// Roberta for Sequence Classification
    #[cfg(feature = "libtorch")]
    Roberta(torch_models::RobertaForSequenceClassification),
    /// XLMRoberta for Sequence Classification
    #[cfg(feature = "libtorch")]
    XLMRoberta(torch_models::RobertaForSequenceClassification),
    /// Albert for Sequence Classification
    #[cfg(feature = "libtorch")]
    Albert(torch_models::AlbertForSequenceClassification),
    /// XLNet for Sequence Classification
    #[cfg(feature = "libtorch")]
    XLNet(torch_models::XLNetForSequenceClassification),
    /// Longformer for Sequence Classification
    #[cfg(feature = "libtorch")]
    Longformer(torch_models::LongformerForSequenceClassification),
    /// ONNX model for Sequence Classification
    #[cfg(feature = "onnx")]
    ONNX(ONNXEncoder),
}

impl ZeroShotClassificationOption {
    /// Instantiate a new zer-shot classification model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `ZeroShotClassificationConfig` - Zero-shot classification pipeline configuration. The type of model created will be inferred from the
    ///   `ModelResources` (Torch or ONNX) and `ModelType` (Architecture for Torch models) variants provided and
    pub fn new(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
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
    fn new_torch(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
        let device: tch::Device = config.device.into();
        let weights_path = config.model_resource.get_torch_local_path()?;
        let mut var_store = VarStore::new(device);
        let model_config =
            &ConfigOption::from_file(config.model_type, config.config_resource.get_local_path()?);
        let model_type = config.model_type;
        let model = match model_type {
            ModelType::Bart => {
                if let ConfigOption::Bart(config) = model_config {
                    Ok(Self::Bart(
                        torch_models::BartForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BartConfig for Bart!".to_string(),
                    ))
                }
            }
            ModelType::Deberta => {
                if let ConfigOption::Deberta(config) = model_config {
                    Ok(Self::Deberta(
                        torch_models::DebertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTa!".to_string(),
                    ))
                }
            }
            ModelType::DebertaV2 => {
                if let ConfigOption::DebertaV2(config) = model_config {
                    Ok(Self::DebertaV2(
                        torch_models::DebertaV2ForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DebertaConfig for DeBERTaV2!".to_string(),
                    ))
                }
            }
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(Self::Bert(
                        torch_models::BertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Bert!".to_string(),
                    ))
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = model_config {
                    Ok(Self::DistilBert(
                        torch_models::DistilBertModelClassifier::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a DistilBertConfig for DistilBert!".to_string(),
                    ))
                }
            }
            ModelType::MobileBert => {
                if let ConfigOption::MobileBert(config) = model_config {
                    Ok(Self::MobileBert(
                        torch_models::MobileBertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a MobileBertConfig for MobileBert!".to_string(),
                    ))
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Roberta(config) = model_config {
                    Ok(Self::Roberta(
                        torch_models::RobertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a RobertaConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = model_config {
                    Ok(Self::XLMRoberta(
                        torch_models::RobertaForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a BertConfig for Roberta!".to_string(),
                    ))
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = model_config {
                    Ok(Self::Albert(
                        torch_models::AlbertForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::XLNet => {
                if let ConfigOption::XLNet(config) = model_config {
                    Ok(Self::XLNet(
                        torch_models::XLNetForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply an AlbertConfig for Albert!".to_string(),
                    ))
                }
            }
            ModelType::Longformer => {
                if let ConfigOption::Longformer(config) = model_config {
                    Ok(Self::Longformer(
                        torch_models::LongformerForSequenceClassification::new(var_store.root(), config)?,
                    ))
                } else {
                    Err(RustBertError::InvalidConfigurationError(
                        "You can only supply a LongformerConfig for Longformer!".to_string(),
                    ))
                }
            }
            #[cfg(feature = "onnx")]
            ModelType::ONNX => Err(RustBertError::InvalidConfigurationError(
                "A `ModelType::ONNX` ModelType was provided in the configuration with `ModelResources::TORCH`, these are incompatible".to_string(),
            )),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Zero shot classification not implemented for {model_type:?}!",
            ))),
        }?;
        var_store.load(weights_path)?;
        cast_var_store(&mut var_store, config.kind, device);
        Ok(model)
    }

    #[cfg(feature = "onnx")]
    pub fn new_onnx(config: &ZeroShotClassificationConfig) -> Result<Self, RustBertError> {
        let onnx_config = ONNXEnvironmentConfig::from_device(config.device);
        let encoder_file = config
            .model_resource
            .get_onnx_local_paths()?
            .encoder_path
            .ok_or(RustBertError::InvalidConfigurationError(
                "An encoder file must be provided for zero-shot classification ONNX models."
                    .to_string(),
            ))?;

        Ok(Self::ONNX(ONNXEncoder::new(encoder_file, &onnx_config)?))
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            #[cfg(feature = "libtorch")]
            Self::Bart(_) => ModelType::Bart,
            #[cfg(feature = "libtorch")]
            Self::Deberta(_) => ModelType::Deberta,
            #[cfg(feature = "libtorch")]
            Self::DebertaV2(_) => ModelType::DebertaV2,
            #[cfg(feature = "libtorch")]
            Self::Bert(_) => ModelType::Bert,
            #[cfg(feature = "libtorch")]
            Self::Roberta(_) => ModelType::Roberta,
            #[cfg(feature = "libtorch")]
            Self::XLMRoberta(_) => ModelType::Roberta,
            #[cfg(feature = "libtorch")]
            Self::DistilBert(_) => ModelType::DistilBert,
            #[cfg(feature = "libtorch")]
            Self::MobileBert(_) => ModelType::MobileBert,
            #[cfg(feature = "libtorch")]
            Self::Albert(_) => ModelType::Albert,
            #[cfg(feature = "libtorch")]
            Self::XLNet(_) => ModelType::XLNet,
            #[cfg(feature = "libtorch")]
            Self::Longformer(_) => ModelType::Longformer,
            #[cfg(feature = "onnx")]
            Self::ONNX(_) => ModelType::ONNX,
        }
    }

    /// Interface method to forward_t() of the particular models.
    pub fn forward_t(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        mask: Option<&ArrayD<i64>>,
        token_type_ids: Option<&ArrayD<i64>>,
        position_ids: Option<&ArrayD<i64>>,
        input_embeds: Option<&ArrayD<f32>>,
        train: bool,
    ) -> ArrayD<f32> {
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
            let token_type_ids_array = token_type_ids;
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let position_ids_array = position_ids;
            #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
            let input_embeds_array = input_embeds;
            let input_ids = to_tensor_i64(input_ids);
            let mask = to_tensor_i64(mask);
            let token_type_ids = to_tensor_i64(token_type_ids);
            let position_ids = to_tensor_i64(position_ids);
            let input_embeds = to_tensor_f32(input_embeds);
            match *self {
                Self::Bart(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids
                                .as_ref()
                                .expect("`input_ids` must be provided for BART models"),
                            mask.as_ref(),
                            None,
                            None,
                            None,
                            train,
                        )
                        .decoder_output,
                )
                .expect("Error converting model output to ndarray"),
                Self::Deberta(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in Deberta forward_t")
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::DebertaV2(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in Deberta V2 forward_t")
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::Bert(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::DistilBert(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in distilbert forward_t")
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::MobileBert(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            None,
                            None,
                            input_embeds.as_ref(),
                            mask.as_ref(),
                            train,
                        )
                        .expect("Error in mobilebert forward_t")
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::Roberta(ref model) | Self::XLMRoberta(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::Albert(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::XLNet(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            None,
                            None,
                            None,
                            token_type_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                Self::Longformer(ref model) => tensor_to_array_f32(
                    &model
                        .forward_t(
                            input_ids.as_ref(),
                            mask.as_ref(),
                            None,
                            token_type_ids.as_ref(),
                            position_ids.as_ref(),
                            input_embeds.as_ref(),
                            train,
                        )
                        .expect("Error in Longformer forward_t")
                        .logits,
                )
                .expect("Error converting model output to ndarray"),
                #[cfg(feature = "onnx")]
                Self::ONNX(ref model) => model
                    .forward(
                        input_ids_array,
                        mask_array,
                        token_type_ids_array,
                        position_ids_array,
                        input_embeds_array,
                    )
                    .expect("Error in ONNX forward pass.")
                    .logits
                    .unwrap(),
            }
        }
        #[cfg(not(feature = "libtorch"))]
        {
            match *self {
                #[cfg(feature = "onnx")]
                Self::ONNX(ref model) => model
                    .forward(input_ids, mask, token_type_ids, position_ids, input_embeds)
                    .expect("Error in ONNX forward pass.")
                    .logits
                    .unwrap(),
                #[cfg(not(feature = "onnx"))]
                _ => unreachable!("no inference backend available"),
            }
        }
    }
}

/// Template used to transform the zero-shot classification labels into a set of
/// natural language hypotheses for natural language inference.
///
/// For example, transform `[positive, negative]` into
/// `[This is a positive review, This is a negative review]`
///
/// The function should take a `&str` as an input and return the formatted String.
///
/// This transformation has a strong impact on the resulting classification accuracy.
/// If no function is provided for zero-shot classification, the default templating
/// function will be used:
///
/// ```rust
/// fn default_template(label: &str) -> String {
///  format!("This example is about {}.", label)
/// }
/// ```
pub type ZeroShotTemplate = Box<dyn Fn(&str) -> String>;

/// Tokenized (input ids, attention masks, token type ids) label pairs.
type TokenizedLabelPairs = (Array2<i64>, Array2<i64>, Array2<i64>);

/// # ZeroShotClassificationModel for Zero Shot Classification
pub struct ZeroShotClassificationModel {
    tokenizer: TokenizerOption,
    zero_shot_classifier: ZeroShotClassificationOption,
}

impl ZeroShotClassificationModel {
    /// Build a new `ZeroShotClassificationModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `SequenceClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    ///
    /// let model = SequenceClassificationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: ZeroShotClassificationConfig,
    ) -> Result<ZeroShotClassificationModel, RustBertError> {
        let vocab_path = config.vocab_resource.get_local_path()?;
        let merges_path = config
            .merges_resource
            .as_ref()
            .map(|resource| resource.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        Self::new_with_tokenizer(config, tokenizer)
    }

    /// Build a new `ZeroShotClassificationModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `config` - `SequenceClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for zero-shot classification.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///  ModelType::Bert,
    ///  "path/to/vocab.txt",
    ///  None,
    ///  false,
    ///  None,
    ///  None,
    /// )?;
    /// let model = SequenceClassificationModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        config: ZeroShotClassificationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<ZeroShotClassificationModel, RustBertError> {
        let zero_shot_classifier = ZeroShotClassificationOption::new(&config)?;

        Ok(ZeroShotClassificationModel {
            tokenizer,
            zero_shot_classifier,
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

    fn prepare_for_model<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_len: usize,
    ) -> Result<TokenizedLabelPairs, RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let label_sentences: Vec<String> = match template {
            Some(function) => labels
                .as_ref()
                .iter()
                .map(|label| function(label))
                .collect(),
            None => labels
                .as_ref()
                .iter()
                .map(|label| format!("This example is about {label}."))
                .collect(),
        };

        let text_pair_list = inputs
            .as_ref()
            .iter()
            .flat_map(|input| {
                label_sentences
                    .iter()
                    .map(move |label_sentence| (*input, label_sentence.as_str()))
            })
            .collect::<Vec<(&str, &str)>>();

        let mut tokenized_input: Vec<TokenizedInput> = self.tokenizer.encode_pair_list(
            text_pair_list.as_ref(),
            max_len,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .ok_or_else(|| RustBertError::ValueError("Got empty iterator as input".to_string()))?;

        let pad_id = self
            .tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for sequence classification should contain a PAD id");
        let mut input_ids = Array2::<i64>::zeros((tokenized_input.len(), max_len));
        let mut token_type_ids = Array2::<i64>::zeros((tokenized_input.len(), max_len));
        for (row, input) in tokenized_input.iter_mut().enumerate() {
            input.token_ids.resize(max_len, pad_id);
            input
                .segment_ids
                .resize(max_len, *input.segment_ids.last().unwrap_or(&0));
            input_ids
                .row_mut(row)
                .assign(&Array1::from(input.token_ids.clone()));
            token_type_ids.row_mut(row).assign(&Array1::from_iter(
                input.segment_ids.iter().map(|&segment| segment as i64),
            ));
        }
        let pad_id = self
            .tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for zero shot classification should contain a PAD id");
        let mask = input_ids.mapv(|value| (value != pad_id) as i64);

        Ok((input_ids, mask, token_type_ids))
    }

    /// Zero shot classification with 1 (and exactly 1) true label.
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to classify.
    /// * `labels` - `&[&str]` Possible labels for the inputs.
    /// * `template` - `Option<Box<dyn Fn(&str) -> String>>` closure to build label propositions. If None, will default to `"This example is {}."`.
    /// * `max_length` -`usize` Maximum sequence length for the inputs. If needed, the input sequence will be truncated before the label template.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Label>, RustBertError>` containing the most likely label for each input sentence or error, if any.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
    ///
    /// let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
    ///
    /// let input_sentence = "Who are you voting for in 2020?";
    /// let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    /// let candidate_labels = &["politics", "public health", "economics", "sports"];
    ///
    /// let output = sequence_classification_model.predict(
    ///  &[input_sentence, input_sequence_2],
    ///  candidate_labels,
    ///  None,
    ///  128,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// outputs:
    /// ```no_run
    /// # use rust_bert::pipelines::sequence_classification::Label;
    /// let output = [
    ///  Label {
    ///      text: "politics".to_string(),
    ///      score: 0.959,
    ///      id: 0,
    ///      sentence: 0,
    ///  },
    ///  Label {
    ///      text: "economy".to_string(),
    ///      score: 0.642,
    ///      id: 2,
    ///      sentence: 1,
    ///  },
    /// ]
    /// .to_vec();
    /// ```
    pub fn predict<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_length: usize,
    ) -> Result<Vec<Label>, RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let num_inputs = inputs.as_ref().len();
        let num_labels = labels.as_ref().len();
        let (input_ids, mask, token_type_ids) =
            self.prepare_for_model(inputs.as_ref(), labels.as_ref(), template, max_length)?;

        let output = self.zero_shot_classifier.forward_t(
            Some(&input_ids.into_dyn()),
            Some(&mask.into_dyn()),
            Some(&token_type_ids.into_dyn()),
            None,
            None,
            false,
        );
        let output = output
            .into_shape_with_order((num_inputs, num_labels, 3))
            .expect("Model output could not be shaped to (inputs, labels, 3)");

        // Entailment probability: softmax over the [contradiction, entailment] logits,
        // renormalized across labels (matching the original single-label pipeline).
        let mut entailment = Array2::<f32>::zeros((num_inputs, num_labels));
        for sentence_idx in 0..num_inputs {
            let label_scores: Vec<f32> = (0..num_labels)
                .map(|label_idx| {
                    let pair = [
                        output[[sentence_idx, label_idx, 0]],
                        output[[sentence_idx, label_idx, 2]],
                    ];
                    softmax_last_dim(&Array1::from(pair.to_vec()).into_dyn())[1]
                })
                .collect();
            let normalized = softmax_last_dim(&Array1::from(label_scores).into_dyn());
            entailment
                .row_mut(sentence_idx)
                .assign(&Array1::from_iter(normalized.iter().cloned()));
        }

        let label_indices = argmax_last_dim(&entailment.clone().into_dyn());

        let mut output_labels: Vec<Label> = vec![];
        for (sentence_idx, &label_index) in label_indices.iter().enumerate() {
            let label_string = labels.as_ref()[label_index as usize].to_string();
            let label = Label {
                text: label_string,
                score: entailment[[sentence_idx, label_index as usize]] as f64,
                id: label_index,
                sentence: sentence_idx,
            };
            output_labels.push(label)
        }
        Ok(output_labels)
    }

    /// Zero shot multi-label classification with 0, 1 or no true label.
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to classify.
    /// * `labels` - `&[&str]` Possible labels for the inputs.
    /// * `template` - `Option<Box<dyn Fn(&str) -> String>>` closure to build label propositions. If None, will default to `"This example is about {}."`.
    /// * `max_length` -`usize` Maximum sequence length for the inputs. If needed, the input sequence will be truncated before the label template.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Vec<Label>>, RustBertError>` containing a vector of labels and their probability for each input text, or error, if any.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
    ///
    /// let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
    ///
    /// let input_sentence = "Who are you voting for in 2020?";
    /// let input_sequence_2 = "The central bank is meeting today to discuss monetary policy.";
    /// let candidate_labels = &["politics", "public health", "economics", "sports"];
    ///
    /// let output = sequence_classification_model.predict_multilabel(
    ///  &[input_sentence, input_sequence_2],
    ///  candidate_labels,
    ///  None,
    ///  128,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    /// outputs:
    /// ```no_run
    /// # use rust_bert::pipelines::sequence_classification::Label;
    /// let output = [
    ///  [
    ///      Label {
    ///          text: "politics".to_string(),
    ///          score: 0.972,
    ///          id: 0,
    ///          sentence: 0,
    ///      },
    ///      Label {
    ///          text: "public health".to_string(),
    ///          score: 0.032,
    ///          id: 1,
    ///          sentence: 0,
    ///      },
    ///      Label {
    ///          text: "economy".to_string(),
    ///          score: 0.006,
    ///          id: 2,
    ///          sentence: 0,
    ///      },
    ///      Label {
    ///          text: "sports".to_string(),
    ///          score: 0.004,
    ///          id: 3,
    ///          sentence: 0,
    ///      },
    ///  ],
    ///  [
    ///      Label {
    ///          text: "politics".to_string(),
    ///          score: 0.975,
    ///          id: 0,
    ///          sentence: 1,
    ///      },
    ///      Label {
    ///          text: "economy".to_string(),
    ///          score: 0.852,
    ///          id: 2,
    ///          sentence: 1,
    ///      },
    ///      Label {
    ///          text: "public health".to_string(),
    ///          score: 0.0818,
    ///          id: 1,
    ///          sentence: 1,
    ///      },
    ///      Label {
    ///          text: "sports".to_string(),
    ///          score: 0.001,
    ///          id: 3,
    ///          sentence: 1,
    ///      },
    ///  ],
    /// ]
    /// .to_vec();
    /// ```
    pub fn predict_multilabel<'a, S, T>(
        &self,
        inputs: S,
        labels: T,
        template: Option<ZeroShotTemplate>,
        max_length: usize,
    ) -> Result<Vec<Vec<Label>>, RustBertError>
    where
        S: AsRef<[&'a str]>,
        T: AsRef<[&'a str]>,
    {
        let num_inputs = inputs.as_ref().len();
        let num_labels = labels.as_ref().len();
        let (input_ids, mask, token_type_ids) =
            self.prepare_for_model(inputs.as_ref(), labels.as_ref(), template, max_length)?;

        let output = self.zero_shot_classifier.forward_t(
            Some(&input_ids.into_dyn()),
            Some(&mask.into_dyn()),
            Some(&token_type_ids.into_dyn()),
            None,
            None,
            false,
        );
        let output = output
            .into_shape_with_order((num_inputs, num_labels, 3))
            .expect("Model output could not be shaped to (inputs, labels, 3)");

        // Entailment probability from the [contradiction, entailment] logit pair.
        let mut entailment = Array2::<f32>::zeros((num_inputs, num_labels));
        for sentence_idx in 0..num_inputs {
            for label_idx in 0..num_labels {
                let pair = [
                    output[[sentence_idx, label_idx, 0]],
                    output[[sentence_idx, label_idx, 2]],
                ];
                entailment[[sentence_idx, label_idx]] =
                    softmax_last_dim(&Array1::from(pair.to_vec()).into_dyn())[1];
            }
        }

        let mut output_labels = vec![];
        for sentence_idx in 0..num_inputs {
            let mut sentence_labels = vec![];

            for (label_index, &score) in entailment.row(sentence_idx).iter().enumerate() {
                let label_string = labels.as_ref()[label_index].to_string();
                let label = Label {
                    text: label_string,
                    score: score as f64,
                    id: label_index as i64,
                    sentence: sentence_idx,
                };
                sentence_labels.push(label);
            }
            output_labels.push(sentence_labels);
        }
        Ok(output_labels)
    }
}
#[cfg(test)]
#[cfg(feature = "libtorch")]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = ZeroShotClassificationConfig::default();
        let _: Box<dyn Send> = Box::new(ZeroShotClassificationModel::new(config));
    }
}
