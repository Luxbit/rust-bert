// Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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

use crate::bart::BartModelOutput;
use crate::common::dropout::Dropout;
pub use crate::mbart::config::{
    MBartConfig, MBartConfigResources, MBartModelResources, MBartVocabResources,
};
use crate::mbart::decoder::MBartDecoder;
use crate::mbart::encoder::MBartEncoder;
use crate::mbart::LayerState;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{
    last_column, option_array2_to_tensor, option_array_to_tensor_f32,
};
use crate::pipelines::generation_utils::{
    Cache, GenerateConfig, GeneratedLogits, LanguageGenerator,
};
use crate::pipelines::translation::Language;
use crate::{Config, RustBertError};
use ndarray::ArrayD;
use std::borrow::Borrow;
use tch::kind::Kind::Int64;
use tch::nn::{embedding, EmbeddingConfig, Init};
use tch::{nn, Tensor};

/// # MBART source languages pre-sets
pub struct MBartSourceLanguages;

/// # MBART target languages pre-sets
pub type MBartTargetLanguages = MBartSourceLanguages;

impl MBartModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart-50-many-to-many-mmt/model",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/rust_model.ot",
    );
}

impl MBartConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart-50-many-to-many-mmt/config",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/config.json",
    );
}

impl MBartVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at <https://github.com/pytorch/fairseq>. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart-50-many-to-many-mmt/vocab",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model",
    );
}

#[rustfmt::skip]
impl MBartSourceLanguages {
    pub const MBART50_MANY_TO_MANY: [Language; 51] = [Language::Arabic, Language::Czech, Language::German, Language::English, Language::Spanish, Language::Estonian, Language::Finnish, Language::French, Language::Gujarati, Language::Hindi, Language::Italian, Language::Japanese, Language::Kazakh, Language::Korean, Language::Lithuanian, Language::Latvian, Language::Burmese, Language::Nepali, Language::Dutch, Language::Romanian, Language::Russian, Language::Sinhala, Language::Turkish, Language::Vietnamese, Language::ChineseMandarin, Language::Afrikaans, Language::Azerbaijani, Language::Bengali, Language::Farsi, Language::Hebrew, Language::Croatian, Language::Indonesian, Language::Georgian, Language::CentralKhmer, Language::Macedonian, Language::Malayalam, Language::Mongolian, Language::Marathi, Language::Polish, Language::Pashto, Language::Portuguese, Language::Swedish, Language::Swahili, Language::Tamil, Language::Thai, Language::Tagalog, Language::Ukrainian, Language::Urdu, Language::Xhosa, Language::Galician, Language::Slovenian];
}

fn _shift_tokens_right(input_ids: &Tensor, pad_token_id: i64) -> Tensor {
    let output = input_ids.masked_fill(&input_ids.eq(-100), pad_token_id);
    let index_eos: Tensor = input_ids
        .ne(pad_token_id)
        .sum_dim_intlist([1].as_slice(), true, Int64)
        - 1;
    output
        .select(1, 0)
        .copy_(&input_ids.gather(1, &index_eos, false).squeeze());
    output
        .slice(1, 1, *output.size().last().unwrap(), 1)
        .copy_(&input_ids.slice(1, 0, *output.size().last().unwrap() - 1, 1));
    output
}

pub struct MBartClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl MBartClassificationHead {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> Result<MBartClassificationHead, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let dense = nn::linear(
            p / "dense",
            config.d_model,
            config.d_model,
            Default::default(),
        );

        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                RustBertError::InvalidConfigurationError(
                    "num_labels not provided in configuration".to_string(),
                )
            })?
            .len() as i64;
        let out_proj = nn::linear(
            p / "out_proj",
            config.d_model,
            num_labels,
            Default::default(),
        );

        let dropout = Dropout::new(config.classifier_dropout.unwrap_or(0.0));

        Ok(MBartClassificationHead {
            dense,
            dropout,
            out_proj,
        })
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        hidden_states
            .apply_t(&self.dropout, train)
            .apply(&self.dense)
            .tanh()
            .apply_t(&self.dropout, train)
            .apply(&self.out_proj)
    }
}

/// # MBart Base model
/// Base architecture for MBart model. Usually complemented with a task-specific head, such as a language model head.
/// It is made of the following blocks:
/// - `encoder`: `MBartEncoder` (transformer) made of a vector of encoding layers
/// - `decoder`: `MBartDecoder` (transformer)  made of a vector of decoding layers with self attention and encoder cross-attention.
///   caching is implemented for the decoder to avoid recalculating static states (encoder key/values and previously calculated decoder key/values)
/// - `pad_token_id`: padding token id
pub struct MBartModel {
    pub(crate) encoder: MBartEncoder,
    decoder: MBartDecoder,
    pub(crate) embeddings: nn::Embedding,
    pad_token_id: i64,
}

impl MBartModel {
    /// Build a new `MBartModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MBart model
    /// * `config` - `MBartConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mbart::{MBartConfig, MBartModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MBartConfig::from_file(config_path);
    /// let mbart: MBartModel = MBartModel::new(&p.root() / "bart", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let pad_token_id = config.pad_token_id.unwrap_or(1);
        let embedding_config = EmbeddingConfig {
            padding_idx: pad_token_id,
            ..Default::default()
        };
        let embeddings: nn::Embedding = embedding(
            p / "shared",
            config.vocab_size,
            config.d_model,
            embedding_config,
        );

        let encoder = MBartEncoder::new(p / "encoder", config);
        let decoder = MBartDecoder::new(p / "decoder", config);

        MBartModel {
            encoder,
            decoder,
            embeddings,
            pad_token_id,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    ///   These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MBartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *hidden_size*) representing the activations of the last decoder hidden state
    ///   - `encoder_hidden_states` - `Option<Tensor>` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state if it was not provided, otherwise None
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::mbart::{MBartConfig, MBartModel};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MBartConfig::from_file(config_path);
    /// # let mbart_model: MBartModel = MBartModel::new(&vs.root(), &config);
    /// let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    /// let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    /// let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    /// let encoder_attention_mask =
    ///  Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    /// let decoder_attention_mask =
    ///  Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    /// let model_output = no_grad(|| {
    ///  mbart_model.forward_t(
    ///      Some(&input_tensor),
    ///      Some(&encoder_attention_mask),
    ///      Some(&target_tensor),
    ///      None,
    ///      Some(&decoder_attention_mask),
    ///      None,
    ///      false,
    ///  )
    /// });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> MBartModelOutput {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() {
            Some(_shift_tokens_right(input_ids.unwrap(), self.pad_token_id))
        } else {
            None
        };

        let decoder_input_ids =
            decoder_input_ids.unwrap_or_else(|| calc_decoder_input_ids.as_ref().unwrap());

        let calc_encoder_output = if encoder_output.is_none() {
            Some(self.encoder.forward_t(
                input_ids.unwrap(),
                attention_mask,
                &self.embeddings,
                train,
            ))
        } else {
            None
        };

        let (calc_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if let Some(calc_encoder_output) = calc_encoder_output {
                (
                    Some(calc_encoder_output.hidden_state),
                    calc_encoder_output.all_hidden_states,
                    calc_encoder_output.all_attentions,
                )
            } else {
                (None, None, None)
            };

        let encoder_output = encoder_output.unwrap_or_else(|| calc_hidden_states.as_ref().unwrap());

        let decoder_output = self.decoder.forward_t(
            decoder_input_ids,
            encoder_output,
            attention_mask,
            decoder_attention_mask,
            &self.embeddings,
            layer_states,
            train,
        );

        MBartModelOutput {
            decoder_output: decoder_output.hidden_state,
            encoder_hidden_state: calc_hidden_states,
            cache: decoder_output.next_decoder_cache,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_decoder_attentions: decoder_output.all_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        }
    }
}

/// # MBart Model for conditional generation
/// MBart model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `MBartModel` Base MBart model
/// - `linear`: Linear layer without bias tied to the weights of the token id embeddings
pub struct MBartForConditionalGeneration {
    base_model: MBartModel,
    final_logits_bias: Tensor,
}

impl MBartForConditionalGeneration {
    /// Build a new `MBartForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MBart model
    /// * `config` - `MBartConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mbart::{MBartConfig, MBartForConditionalGeneration};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MBartConfig::from_file(config_path);
    /// let mbart: MBartForConditionalGeneration =
    ///  MBartForConditionalGeneration::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartForConditionalGeneration
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = MBartModel::new(p / "model", config);
        let final_logits_bias = p.var(
            "final_logits_bias",
            &[1, config.vocab_size],
            Init::Const(0.0),
        );

        MBartForConditionalGeneration {
            base_model,
            final_logits_bias,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    ///   These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MBartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocabulary item and position
    ///   - `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::mbart::{MBartConfig, MBartForConditionalGeneration};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MBartConfig::from_file(config_path);
    /// # let mbart_model: MBartForConditionalGeneration = MBartForConditionalGeneration::new(&vs.root(), &config);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
    ///    mbart_model
    ///      .forward_t(Some(&input_tensor),
    ///                 Some(&encoder_attention_mask),
    ///                 None,
    ///                 Some(&target_tensor),
    ///                 Some(&decoder_attention_mask),
    ///                 None,
    ///                 false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> MBartModelOutput {
        let base_model_output = self.base_model.forward_t(
            input_ids,
            attention_mask,
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            old_layer_states,
            train,
        );

        let lm_logits = base_model_output
            .decoder_output
            .linear::<Tensor>(&self.base_model.embeddings.ws, None)
            + &self.final_logits_bias;
        BartModelOutput {
            decoder_output: lm_logits,
            ..base_model_output
        }
    }

    pub fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        self.base_model
            .encoder
            .forward_t(
                input_ids,
                attention_mask,
                &self.base_model.embeddings,
                false,
            )
            .hidden_state
    }
}

/// # MBart Model for sequence classification
/// MBart model with a classification head
/// It is made of the following blocks:
/// - `base_model`: `MBartModel` Base MBart model
/// - `classification_head`: `BartClassificationHead` made of 2 linear layers mapping hidden states to a target class
/// - `eos_token_id`: token id for the EOS token carrying the pooled representation for classification
pub struct MBartForSequenceClassification {
    base_model: MBartModel,
    classification_head: MBartClassificationHead,
    eos_token_id: i64,
}

impl MBartForSequenceClassification {
    /// Build a new `MBartForSequenceClassification`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the MBart model
    /// * `config` - `MBartConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::mbart::{MBartConfig, MBartForSequenceClassification};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = MBartConfig::from_file(config_path);
    /// let mbart: MBartForSequenceClassification =
    ///  MBartForSequenceClassification::new(&p.root(), &config).unwrap();
    /// ```
    pub fn new<'p, P>(
        p: P,
        config: &MBartConfig,
    ) -> Result<MBartForSequenceClassification, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let base_model = MBartModel::new(p / "model", config);
        let classification_head = MBartClassificationHead::new(p / "classification_head", config)?;
        let eos_token_id = config.eos_token_id.unwrap_or(3);
        Ok(MBartForSequenceClassification {
            base_model,
            classification_head,
            eos_token_id,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    ///   These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `MBartModelOutput` containing:
    ///   - `decoder_output` - `Tensor` of shape (*batch size*, *num_classes*) representing the activations for each class and batch item
    ///   - `encoder_hidden_states` - `Option<Tensor>` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state if it was not provided, otherwise None.
    ///   - `cache` - `(Option<Tensor>, Option<Vec<&LayerState, &LayerState>>)` of length *n_layer* containing the encoder padding mask and past keys and values for both the self attention and the encoder cross attention of each layer of the decoder.
    ///   - `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    ///   - `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///   - `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::mbart::{MBartConfig, MBartForSequenceClassification};
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = MBartConfig::from_file(config_path);
    /// # let mbart_model: MBartForSequenceClassification = MBartForSequenceClassification::new(&vs.root(), &config).unwrap();
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let model_output = no_grad(|| {
    ///    mbart_model
    ///      .forward_t(&input_tensor,
    ///                 Some(&encoder_attention_mask),
    ///                 None,
    ///                 Some(&target_tensor),
    ///                 Some(&decoder_attention_mask),
    ///                 false)
    ///    });
    /// ```
    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> MBartModelOutput {
        let base_model_output = self.base_model.forward_t(
            Some(input_ids),
            attention_mask,
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            None,
            train,
        );
        let eos_mask = input_ids.eq(self.eos_token_id);
        let reshape = eos_mask.sum_dim_intlist([1].as_slice(), true, Int64);
        let sentence_representation = base_model_output
            .decoder_output
            .permute([2, 0, 1])
            .masked_select(&eos_mask)
            .view((-1, reshape.size()[0] * reshape.int64_value(&[0, 0])))
            .transpose(0, 1)
            .view((
                base_model_output.decoder_output.size()[0],
                -1,
                *base_model_output.decoder_output.size().last().unwrap(),
            ))
            .select(1, -1);

        let logits = self
            .classification_head
            .forward_t(&sentence_representation, train);
        MBartModelOutput {
            decoder_output: logits,
            encoder_hidden_state: base_model_output.encoder_hidden_state,
            cache: None,
            all_decoder_hidden_states: base_model_output.all_decoder_hidden_states,
            all_decoder_attentions: base_model_output.all_decoder_attentions,
            all_encoder_hidden_states: base_model_output.all_encoder_hidden_states,
            all_encoder_attentions: base_model_output.all_encoder_attentions,
        }
    }
}

/// Container holding a MBART model output
pub type MBartModelOutput = BartModelOutput;

/// # Language generation model based on the MBart architecture
pub struct MBartGenerator {
    model: MBartForConditionalGeneration,
    tokenizer: TokenizerOption,
    var_store: nn::VarStore,
    generate_config: GenerateConfig,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    forced_eos_token_id: Option<i64>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: i64,
}

impl MBartGenerator {
    /// Build a new `MBartGenerator`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `merges_path` - Path to the bpe merges, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `config_path` - Path to the model configuration, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `weights_path` - Path to the model weight files. These need to be converted form the `.bin` to `.ot` format using the utility script provided.
    /// * `device` - Device to run the model on, e.g. `Device::Cpu` or `Device::Cuda(0)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::path::PathBuf;
    /// # use tch::Device;
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::mbart::MBartGenerator;
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// # let mut home: PathBuf = dirs::home_dir().unwrap();
    /// # home.push("rustbert");
    /// # home.push("openai-gpt");
    /// # let config_path = &home.as_path().join("config.json");
    /// # let vocab_path = &home.as_path().join("vocab.txt");
    /// # let merges_path = &home.as_path().join("merges.txt");
    /// # let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::cuda_if_available();
    /// let generate_config = GenerateConfig {
    ///  max_length: Some(30),
    ///  do_sample: true,
    ///  num_beams: 5,
    ///  temperature: 1.1,
    ///  num_return_sequences: 3,
    ///  ..Default::default()
    /// };
    /// let mbart_generator = MBartGenerator::new(generate_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(generate_config: GenerateConfig) -> Result<MBartGenerator, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;

        let tokenizer = TokenizerOption::from_file(
            ModelType::MBart,
            vocab_path.to_str().unwrap(),
            None,
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
    ) -> Result<MBartGenerator, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let device = generate_config.device;

        generate_config.validate();
        let mut var_store = nn::VarStore::new(device.into());

        let config = MBartConfig::from_file(config_path);
        let model = MBartForConditionalGeneration::new(var_store.root(), &config);
        crate::resources::load_weights(
            &generate_config.model_resource,
            &mut var_store,
            generate_config.kind,
            device.into(),
        )?;

        let bos_token_id = Some(config.bos_token_id.unwrap_or(0));
        let eos_token_ids = Some(match config.eos_token_id {
            Some(value) => vec![value],
            None => vec![2],
        });
        let forced_eos_token_id = config.forced_eos_token_id;
        let pad_token_id = Some(config.pad_token_id.unwrap_or(1));
        let vocab_size = config.vocab_size;
        let is_encoder_decoder = true;
        let decoder_start_id = config.decoder_start_token_id;
        let max_position_embeddings = config.max_position_embeddings;

        Ok(MBartGenerator {
            model,
            tokenizer,
            var_store,
            generate_config,
            bos_token_id,
            eos_token_ids,
            forced_eos_token_id,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
        })
    }
}

impl PrivateLanguageGenerator for MBartGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_device(&self) -> crate::Device {
        crate::Device::from(self.var_store.device())
    }
    #[cfg(feature = "libtorch")]
    #[cfg(feature = "libtorch")]
    fn get_var_store_mut(&mut self) -> Result<&mut nn::VarStore, RustBertError> {
        Ok(&mut self.var_store)
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_forced_eos_token_id(&self) -> Option<i64> {
        self.forced_eos_token_id
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }

    fn forward_t(
        &self,
        input_ids: Option<&ndarray::Array2<i64>>,
        cache: Cache,
        attention_mask: Option<&ndarray::Array2<i64>>,
        _token_type_ids: Option<&ndarray::Array2<i64>>,
        _position_ids: Option<&ndarray::Array2<i64>>,
        _input_embeds: Option<&ndarray::ArrayD<f32>>,
        encoder_outputs: Option<&ndarray::ArrayD<f32>>,
        decoder_input_ids: Option<&ndarray::Array2<i64>>,
        train: bool,
    ) -> Result<GeneratedLogits, RustBertError> {
        #[cfg(feature = "libtorch")]
        let (
            input_ids,
            attention_mask,
            _token_type_ids,
            _position_ids,
            _input_embeds,
            encoder_outputs,
            decoder_input_ids,
        ) = (
            option_array2_to_tensor(input_ids),
            option_array2_to_tensor(attention_mask),
            option_array2_to_tensor(_token_type_ids),
            option_array2_to_tensor(_position_ids),
            option_array_to_tensor_f32(_input_embeds),
            option_array_to_tensor_f32(encoder_outputs),
            option_array2_to_tensor(decoder_input_ids),
        );
        #[cfg(not(feature = "libtorch"))]
        let (_, _, _, _, _, _, _) = (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            encoder_outputs,
            decoder_input_ids,
        );
        let base_model_output = match cache {
            Cache::BARTCache(cached_layer_states) => self.model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                encoder_outputs.as_ref(),
                decoder_input_ids.as_ref(),
                None,
                cached_layer_states,
                train,
            ),

            Cache::None => self.model.forward_t(
                input_ids.as_ref(),
                attention_mask.as_ref(),
                encoder_outputs.as_ref(),
                decoder_input_ids.as_ref(),
                None,
                None,
                train,
            ),
            _ => {
                return Err(RustBertError::ValueError(
                    "Cache not compatible with MBART Model".into(),
                ));
            }
        };

        Ok(GeneratedLogits {
            lm_logits: crate::common::tensor_conversion::tensor_to_array_f32(
                &base_model_output.decoder_output,
            )?,
            cache: Cache::BARTCache(base_model_output.cache),
        })
    }

    fn get_max_positions_embeddings(&self) -> Option<i64> {
        Some(self.max_position_embeddings)
    }

    fn encode(
        &self,
        input_ids: &ndarray::Array2<i64>,
        attention_mask: Option<&ndarray::Array2<i64>>,
    ) -> Option<ArrayD<f32>> {
        #[cfg(feature = "libtorch")]
        {
            let input_ids_tensor = option_array2_to_tensor(Some(input_ids))?;
            let attention_mask_tensor = option_array2_to_tensor(attention_mask);
            let encoded = self
                .model
                .encode(&input_ids_tensor, attention_mask_tensor.as_ref());
            Some(
                crate::common::tensor_conversion::tensor_to_array_f32(&encoded)
                    .expect("Error converting encoder output"),
            )
        }
        #[cfg(not(feature = "libtorch"))]
        {
            None
        }
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: ndarray::Array2<i64>,
        encoder_outputs: Option<&'a ArrayD<f32>>,
        past: Cache,
        attention_mask: ndarray::Array2<i64>,
    ) -> PreparedInput<'a> {
        match past {
            Cache::BARTCache(past) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(last_column(&input_ids)),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(past),
            },
            Cache::None => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::BARTCache(None),
            },
            _ => panic!("Cache type incompatible with MBart"),
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<ArrayD<f32>>,
        beam_indices: &[i64],
    ) -> Option<ArrayD<f32>> {
        #[cfg(feature = "libtorch")]
        let beam_indices = Tensor::from_slice(beam_indices);
        #[cfg(feature = "libtorch")]
        let encoder_outputs_tensor = encoder_outputs.map(|output| {
            crate::common::tensor_conversion::array_to_tensor_f32(&output)
                .expect("Error converting encoder output")
        });
        let encoder_outputs = encoder_outputs_tensor.map(|value| {
            crate::common::tensor_conversion::tensor_to_array_f32(
                &value.index_select(0, &beam_indices),
            )
            .expect("Error converting encoder output")
        });
        match past {
            Cache::BARTCache(old_cache_option) => {
                if let Some(old_cache) = old_cache_option {
                    for (self_layer_state, encoder_layer_state) in old_cache.iter_mut() {
                        if self_layer_state.is_some() {
                            self_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(&beam_indices)
                        };
                        if encoder_layer_state.is_some() {
                            encoder_layer_state
                                .as_mut()
                                .unwrap()
                                .reorder_cache(&beam_indices)
                        };
                    }
                }
            }
            Cache::None => {}
            _ => {
                panic!("Invalid cache for MBart model");
            }
        };
        encoder_outputs
    }
}

impl LanguageGenerator for MBartGenerator {}

#[cfg(test)]
mod test {
    use tch::Device;

    use crate::{
        resources::{RemoteResource, ResourceProvider},
        Config,
    };

    use super::*;

    #[test]
    #[ignore] // compilation is enough, no need to run
    fn mbart_model_send() {
        let config_resource = Box::new(RemoteResource::from_pretrained(
            MBartConfigResources::MBART50_MANY_TO_MANY,
        ));
        let config_path = config_resource.get_local_path().expect("");

        //    Set-up masked LM model
        let device = Device::cuda_if_available();
        let vs = tch::nn::VarStore::new(device);
        let config = MBartConfig::from_file(config_path);

        let _: Box<dyn Send> = Box::new(MBartModel::new(vs.root(), &config));
    }
}
