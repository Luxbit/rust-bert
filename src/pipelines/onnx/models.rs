use crate::pipelines::common::{ConfigOption, TokenizerOption};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput};
use crate::pipelines::onnx::config::ONNXEnvironmentConfig;
use crate::pipelines::onnx::decoder::ONNXDecoder;
use crate::pipelines::onnx::encoder::ONNXEncoder;
use crate::{Config, RustBertError};

#[cfg(feature = "libtorch")]
use crate::common::tensor_conversion::{
    array_to_tensor_f32, tensor_to_array_f32, tensor_to_array_i64, tensor_to_vec_i64,
};
#[cfg(feature = "libtorch")]
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
#[cfg(feature = "libtorch")]
use crate::pipelines::generation_utils::LanguageGenerator;

use ndarray::ArrayD;
use ort::session::SessionOutputs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// # ONNX Model configuration
/// Represents a shared subset of commonly used model parameters used for text generation
/// or classifications tasks. Note that pipelines are compatible with the use of an ONNXModel
/// matched with the corresponding model configuration (e.g. an ONNX exported BERT model would
/// be compatible in pipelines with a `BertConfig`).
///
/// This is provided for extending the support of models in pipelines to models that have
/// not yet been ported to the Torch version in this crate.
///
/// The fields for this configuration are described at
/// https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig
pub struct ONNXModelConfig {
    pub bos_token_id: Option<i64>,
    pub eos_token_ids: Option<Vec<i64>>,
    pub pad_token_id: Option<i64>,
    pub forced_bos_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
    pub vocab_size: i64,
    pub decoder_start_token_id: Option<i64>,
    pub max_position_embeddings: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
}

impl Config for ONNXModelConfig {}

/// # ONNX Causal Generator
/// Container for an ONNX decoder model and the corresponding sessions. This model can be used for
/// causal generation or prefix language models. It may contain one or two sessions to handle the
/// initial generation stage (no cached key values present) and subsequent generation stages (cached
/// keys and values are available from the previous token generated, avoiding unnecessary re-computation).
///
/// The recommended instantiation is done via the `new` and `new_with_tokenizer` methods.
pub struct ONNXCausalGenerator {
    decoder_without_past: Option<ONNXDecoder>,
    decoder_with_past: Option<ONNXDecoder>,
    generate_config: GenerateConfig,
    tokenizer: TokenizerOption,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    forced_bos_token_id: Option<i64>,
    forced_eos_token_id: Option<i64>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: Option<i64>,
    use_past: bool,
}

impl ONNXCausalGenerator {
    /// Create a new `ONNXCausalGenerator` from a `GenerateConfig`.
    ///
    /// Extract the required model, tokenizer and configuration resources from a `GenerateConfig`.
    /// Note that the `model_resources` field of the `GenerateConfig` provided should be of the
    /// `ModelResources::ONNX` type, passing a `ModelResources::Torch` resource will cause the model
    /// to fail.
    ///
    /// The tokenizer is automatically created based on the `model_type` field of the `GenerateConfig`.
    /// In order to create an `ONNXCausalGenerator` the user must therefore provide an actual (non-ONNX)
    /// `model_type` paired with a set of ONNX resources.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXCausalGenerator;
    /// use rust_bert::resources::RemoteResource;
    /// let generate_config = GenerateConfig {
    ///  model_type: ModelType::GPT2,
    ///  model_resource: ModelResource::ONNX(ONNXModelResources {
    ///      encoder_resource: None,
    ///      decoder_resource: Some(Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/gpt2/resolve/main/decoder_model.onnx",
    ///          "onnx-gpt2",
    ///      ))),
    ///      decoder_with_past_resource: Some(Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/gpt2/resolve/main/decoder_with_past_model.onnx",
    ///          "onnx-gpt2",
    ///      ))),
    ///  }),
    ///  config_resource: Box::new(RemoteResource::new(
    ///      "https://huggingface.co/optimum/gpt2/resolve/main/config.json",
    ///      "onnx-gpt2",
    ///  )),
    ///  vocab_resource: Box::new(RemoteResource::new(
    ///      "https://huggingface.co/gpt2/resolve/main/vocab.json",
    ///      "onnx-gpt2",
    ///  )),
    ///  merges_resource: Some(Box::new(RemoteResource::new(
    ///      "https://huggingface.co/gpt2/resolve/main/merges.txt",
    ///      "onnx-gpt2",
    ///  ))),
    ///  ..Default::default()
    /// };
    /// let onnx_config = Some(ONNXEnvironmentConfig::default());
    /// let onnx_causal_generator =
    ///  ONNXCausalGenerator::new(generate_config, onnx_config.as_ref())
    ///      .unwrap();
    /// ```
    pub fn new(
        generate_config: GenerateConfig,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .map(|r| r.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            generate_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_ref().and_then(|path| path.to_str()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer, onnx_config)
    }

    /// Create a new `ONNXCausalGenerator` from a `GenerateConfig` and `TokenizerOption`.
    ///
    /// Extract the required model and configuration resources from a `GenerateConfig`.
    /// Note that the `model_resources` field of the `GenerateConfig` provided should be of the
    /// `ModelResources::ONNX` type, passing a `ModelResources::Torch` resource will cause the model
    /// to fail.
    ///
    /// A tokenizer must be provided by the user and can be customized to use non-default settings.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::common::{
    ///  ModelResource, ModelType, ONNXModelResources, TokenizerOption,
    /// };
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXCausalGenerator;
    /// use rust_bert::resources::RemoteResource;
    /// let generate_config = GenerateConfig {
    ///  model_resource: ModelResource::ONNX(ONNXModelResources {
    ///      encoder_resource: None,
    ///      decoder_resource: Some(Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/gpt2/resolve/main/decoder_model.onnx",
    ///          "onnx-gpt2",
    ///      ))),
    ///      decoder_with_past_resource: Some(Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/gpt2/resolve/main/decoder_with_past_model.onnx",
    ///          "onnx-gpt2",
    ///      ))),
    ///  }),
    ///  config_resource: Box::new(RemoteResource::new(
    ///      "https://huggingface.co/optimum/gpt2/resolve/main/config.json",
    ///      "onnx-gpt2",
    ///  )),
    ///  ..Default::default()
    /// };
    /// let onnx_config = Some(ONNXEnvironmentConfig::default());
    /// let lower_case = false;
    /// let strip_accents = None;
    /// let add_prefix_space = None;
    /// let tokenizer = TokenizerOption::from_file(
    ///  ModelType::GPT2,
    ///  "path/to/vocab.json",
    ///  Some("path/to/merges.txt"),
    ///  lower_case,
    ///  strip_accents,
    ///  add_prefix_space,
    /// )
    /// .unwrap();
    /// let onnx_causal_generator = ONNXCausalGenerator::new_with_tokenizer(
    ///  generate_config,
    ///  tokenizer,
    ///  onnx_config.as_ref(),
    /// )
    /// .unwrap();
    /// ```
    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let model_config = ConfigOption::from_file(generate_config.model_type, config_path);

        let onnx_local_paths = generate_config.model_resource.get_onnx_local_paths()?;
        let (decoder_without_past_file, decoder_with_past_file) = (
            onnx_local_paths.decoder_path,
            onnx_local_paths.decoder_with_past_path,
        );

        if decoder_without_past_file.is_none() & decoder_with_past_file.is_none() {
            return Err(RustBertError::InvalidConfigurationError("Must provide at least one of `decoder_without_past_file`, `decoder_with_past_file`, both set to None".to_string()));
        }

        let default_onnx_config;
        let onnx_config = match onnx_config {
            Some(onnx_config) => onnx_config,
            None => {
                default_onnx_config =
                    ONNXEnvironmentConfig::from_device(crate::Device::from(generate_config.device));
                &default_onnx_config
            }
        };

        let decoder_without_past = if let Some(model_file) = decoder_without_past_file {
            Some(ONNXDecoder::new(model_file, true, onnx_config)?)
        } else {
            None
        };
        let decoder_with_past = if let Some(model_file) = decoder_with_past_file {
            Some(ONNXDecoder::new(model_file, true, onnx_config)?)
        } else {
            None
        };

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let forced_bos_token_id = model_config.get_forced_bos_token_id();
        let forced_eos_token_id = model_config.get_forced_eos_token_id();
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = model_config.get_max_len();
        let is_encoder_decoder = false;
        let vocab_size = model_config.get_vocab_size();
        let decoder_start_id = model_config.get_decoder_start_token_id();
        let use_past = decoder_with_past.is_some();

        Ok(Self {
            decoder_without_past,
            decoder_with_past,
            generate_config,
            tokenizer,
            bos_token_id,
            eos_token_ids,
            forced_bos_token_id,
            forced_eos_token_id,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
            use_past,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input array of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `encoder_hidden_states` - Optional array of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*). These correspond to the encoder last hidden state.
    /// * `encoder_attention_mask` - Optional attention mask for the encoder outputs. Positions with a mask with value 0 will be masked.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented starting from the length of the past input.
    /// * `layer_states` - Optional `Cache` container containing the past keys and values. When provided, these are concatenated with the current input keys and values.
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - logits of shape (*batch size*, *sequence_length*, *vocab_size*) representing the activations of the last hidden state
    ///   - `cache` - `Cache`  containing the past keys and values of each layer.
    pub fn forward(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        attention_mask: Option<&ArrayD<i64>>,
        encoder_hidden_states: Option<&ArrayD<f32>>,
        encoder_attention_mask: Option<&ArrayD<i64>>,
        position_ids: Option<&ArrayD<i64>>,
        layer_states: Option<&Cache>,
    ) -> Result<LMModelOutput, RustBertError> {
        match (
            &self.decoder_without_past,
            &self.decoder_with_past,
            layer_states,
        ) {
            (Some(ref decoder_without_past), _, None)
            | (Some(ref decoder_without_past), _, Some(Cache::None)) => decoder_without_past
                .forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    position_ids,
                    None,
                ),
            (_, Some(ref decoder_with_past), Some(Cache::ONNXCache(ref onnx_cache))) => {
                decoder_with_past.forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    position_ids,
                    Some(onnx_cache),
                )
            }
            (Some(ref decoder_without_past), None, Some(Cache::ONNXCache(_))) => {
                decoder_without_past.forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    position_ids,
                    None,
                )
            }
            (None, _, None) => Err(RustBertError::ValueError(
                "No decoder_without_cache loaded and no cache provided.".to_string(),
            )),
            (None, None, _) => Err(RustBertError::ValueError(
                "No decoder provided.".to_string(),
            )),
            (_, _, Some(cache)) => Err(RustBertError::ValueError(format!(
                "Invalid cache type provided, expected Cache::ONNXLayerCache, got {:?}.",
                cache
            ))),
        }
    }
}

#[cfg(feature = "libtorch")]
impl PrivateLanguageGenerator for ONNXCausalGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_device(&self) -> tch::Device {
        tch::Device::Cpu
    }
    fn get_var_store_mut(&mut self) -> Result<&mut tch::nn::VarStore, RustBertError> {
        Err(RustBertError::ValueError(
            "No VarStore available for ONNX models".to_string(),
        ))
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
    fn get_forced_bos_token_id(&self) -> Option<i64> {
        self.forced_bos_token_id
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
    fn get_max_positions_embeddings(&self) -> Option<i64> {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&tch::Tensor>,
        layer_past: Cache,
        attention_mask: Option<&tch::Tensor>,
        _token_type_ids: Option<&tch::Tensor>,
        position_ids: Option<&tch::Tensor>,
        _input_embeds: Option<&tch::Tensor>,
        _encoder_outputs: Option<&tch::Tensor>,
        _decoder_input_ids: Option<&tch::Tensor>,
        _train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let input_ids = input_ids.map(tensor_to_array_i64).transpose()?;
        let attention_mask = attention_mask.map(tensor_to_array_i64).transpose()?;
        let position_ids = position_ids.map(tensor_to_array_i64).transpose()?;
        self.forward(
            input_ids.as_ref(),
            attention_mask.as_ref(),
            None,
            None,
            position_ids.as_ref(),
            Some(&layer_past),
        )
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: tch::Tensor,
        _encoder_outputs: Option<&'a tch::Tensor>,
        past: Cache,
        attention_mask: tch::Tensor,
    ) -> PreparedInput<'a> {
        let position_ids = (attention_mask
            .totype(tch::Kind::Int64)
            .cumsum(-1, tch::Kind::Int64)
            - 1)
        .masked_fill(&attention_mask.eq(0), 1);

        match (past, self.use_past) {
            (Cache::ONNXCache(past), true) => PreparedInput {
                prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids.select(1, -1).unsqueeze(-1)),
                prepared_past: Cache::ONNXCache(past),
            },
            _ => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids),
                prepared_past: Cache::None,
            },
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<tch::Tensor>,
        beam_indices: &tch::Tensor,
    ) -> Option<tch::Tensor> {
        let beam_indices = tensor_to_vec_i64(beam_indices).expect("Error converting beam indices");
        match past {
            Cache::ONNXCache(cached_decoder_state) => {
                for layer_past in cached_decoder_state.values.values_mut() {
                    *layer_past = crate::common::tensor_ops::gather_rows(layer_past, &beam_indices);
                }
                None
            }
            Cache::None => None,
            _ => {
                panic!("Invalid cache for ONNX model");
            }
        }
    }
}

#[cfg(feature = "libtorch")]
impl LanguageGenerator for ONNXCausalGenerator {}

/// # ONNX Conditional Generator
/// Container for an ONNX encoder/decoder model and the corresponding sessions. This model can be used for
/// conditional language models. It may contain two or three sessions to handle the encoding,
/// initial generation stage (no cached key values present) and optionally subsequent generation stages (cached
/// keys and values are available from the previous token generated, avoiding unnecessary re-computation).
///
/// The recommended instantiation is done via the `new` and `new_with_tokenizer` methods.
pub struct ONNXConditionalGenerator {
    encoder: ONNXEncoder,
    decoder_without_past: Option<ONNXDecoder>,
    decoder_with_past: Option<ONNXDecoder>,
    generate_config: GenerateConfig,
    tokenizer: TokenizerOption,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    forced_bos_token_id: Option<i64>,
    forced_eos_token_id: Option<i64>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: Option<i64>,
    use_past: bool,
}

impl ONNXConditionalGenerator {
    /// Create a new `ONNXConditionalGenerator` from a `GenerateConfig`.
    ///
    /// Extract the required model, tokenizer and configuration resources from a `GenerateConfig`.
    /// Note that the `model_resources` field of the `GenerateConfig` provided should be of the
    /// `ModelResources::ONNX` type, passing a `ModelResources::Torch` resource will cause the model
    /// to fail.
    ///
    /// The tokenizer is automatically created based on the `model_type` field of the `GenerateConfig`.
    /// In order to create an `ONNXConditionalGenerator` the user must therefore provide an actual (non-ONNX)
    /// `model_type` paired with a set of ONNX resources.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXConditionalGenerator;
    /// use rust_bert::resources::RemoteResource;
    /// let generate_config = GenerateConfig {
    ///  model_type: ModelType::M2M100,
    ///  model_resource: ModelResource::ONNX(ONNXModelResources {
    ///           encoder_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/encoder_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///          decoder_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///          decoder_with_past_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_with_past_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///      }),
    ///      config_resource: Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/m2m100_418M/resolve/main/config.json",
    ///          "onnx-m2m100_418M",
    ///      )),
    ///      vocab_resource: Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/m2m100_418M/resolve/main/vocab.json",
    ///          "onnx-m2m100_418M",
    ///      )),
    ///      merges_resource: Some(Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/m2m100_418M/resolve/main/sentencepiece.bpe.model",
    ///          "onnx-m2m100_418M",
    ///      ))),
    ///  ..Default::default()
    /// };
    /// let onnx_config = Some(ONNXEnvironmentConfig::default());
    /// let onnx_conditional_generator =
    ///  ONNXConditionalGenerator::new(generate_config, onnx_config.as_ref())
    ///      .unwrap();
    /// ```
    pub fn new(
        generate_config: GenerateConfig,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .map(|r| r.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            generate_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_ref().and_then(|path| path.to_str()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer, onnx_config)
    }

    /// Create a new `ONNXConditionalGenerator` from a `GenerateConfig` and `TokenizerOption`.
    ///
    /// Extract the required model and configuration resources from a `GenerateConfig`.
    /// Note that the `model_resources` field of the `GenerateConfig` provided should be of the
    /// `ModelResources::ONNX` type, passing a `ModelResources::Torch` resource will cause the model
    /// to fail.
    ///
    /// A tokenizer must be provided by the user and can be customized to use non-default settings.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::common::{
    ///  ModelResource, ModelType, ONNXModelResources, TokenizerOption,
    /// };
    /// use rust_bert::pipelines::generation_utils::GenerateConfig;
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXConditionalGenerator;
    /// use rust_bert::resources::RemoteResource;
    /// let generate_config = GenerateConfig {
    ///  model_resource: ModelResource::ONNX(ONNXModelResources {
    ///           encoder_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/encoder_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///          decoder_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///          decoder_with_past_resource: Some(Box::new(RemoteResource::new(
    ///              "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_with_past_model.onnx",
    ///              "onnx-m2m100_418M",
    ///          ))),
    ///      }),
    ///      config_resource: Box::new(RemoteResource::new(
    ///          "https://huggingface.co/optimum/m2m100_418M/resolve/main/config.json",
    ///          "onnx-m2m100_418M",
    ///      )),
    ///  ..Default::default()
    /// };
    /// let onnx_config = Some(ONNXEnvironmentConfig::default());
    /// let lower_case = false;
    /// let strip_accents = None;
    /// let add_prefix_space = None;
    /// let tokenizer = TokenizerOption::from_file(
    ///  ModelType::M2M100,
    ///  "path/to/vocab.json",
    ///  Some("path/to/merges.txt"),
    ///  lower_case,
    ///  strip_accents,
    ///  add_prefix_space,
    /// )
    /// .unwrap();
    /// let onnx_conditional_generator = ONNXConditionalGenerator::new_with_tokenizer(
    ///  generate_config,
    ///  tokenizer,
    ///  onnx_config.as_ref(),
    /// )
    /// .unwrap();
    /// ```
    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let model_config = ConfigOption::from_file(generate_config.model_type, config_path);

        let onnx_local_paths = generate_config.model_resource.get_onnx_local_paths()?;
        let (encoder_file, decoder_without_past_file, decoder_with_past_file) = (
            onnx_local_paths.encoder_path,
            onnx_local_paths.decoder_path,
            onnx_local_paths.decoder_with_past_path,
        );

        if decoder_without_past_file.is_none() & decoder_with_past_file.is_none() {
            return Err(RustBertError::InvalidConfigurationError("Must provide at least one of `decoder_without_past_file`, `decoder_with_past_file`, both set to None".to_string()));
        }

        let default_onnx_config;
        let onnx_config = match onnx_config {
            Some(onnx_config) => onnx_config,
            None => {
                default_onnx_config =
                    ONNXEnvironmentConfig::from_device(crate::Device::from(generate_config.device));
                &default_onnx_config
            }
        };

        let encoder_file = encoder_file.ok_or(RustBertError::InvalidConfigurationError(format!("ONNXConditionalGenerator requires an `encoder_path` to be provided in the `ModelResources`, got {:?}", generate_config.model_resource)))?;

        let encoder = ONNXEncoder::new(encoder_file, onnx_config)?;
        let decoder_without_past = if let Some(model_file) = decoder_without_past_file {
            Some(ONNXDecoder::new(model_file, true, onnx_config)?)
        } else {
            None
        };
        let decoder_with_past = if let Some(model_file) = decoder_with_past_file {
            Some(ONNXDecoder::new(model_file, true, onnx_config)?)
        } else {
            None
        };

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let forced_bos_token_id = model_config.get_forced_bos_token_id();
        let forced_eos_token_id = model_config.get_forced_eos_token_id();
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = model_config.get_max_len();
        let is_encoder_decoder = true;
        let vocab_size = model_config.get_vocab_size();
        let decoder_start_id = model_config.get_decoder_start_token_id();
        let use_past = decoder_with_past.is_some();

        Ok(Self {
            encoder,
            decoder_without_past,
            decoder_with_past,
            generate_config,
            tokenizer,
            bos_token_id,
            eos_token_ids,
            forced_bos_token_id,
            forced_eos_token_id,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
            use_past,
        })
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input array of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` -  Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_hidden_states` - Optional array of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*). These correspond to the encoder last hidden state.
    /// * `encoder_attention_mask` - Optional mask of shape (*batch size*, *sequence_length*) for the encoder hidden states. Masked position have value 0, non-masked value 1. If None set to 1
    /// * `decoder_input_ids` - Optional input array of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialized with a BOS token)
    /// * `layer_states` - Optional `Cache` container containing the past keys and values. When provided, these are concatenated with the current input keys and values.
    ///
    /// # Returns
    ///
    /// * `LMModelOutput` containing:
    ///   - `lm_logits` - logits of shape (*batch size*, *sequence_length*, *vocab_size*) representing the activations of the last hidden state
    ///   - `cache` - `Cache`  containing the past keys and values of each layer.
    pub fn forward(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        attention_mask: Option<&ArrayD<i64>>,
        encoder_hidden_states: Option<&ArrayD<f32>>,
        encoder_attention_mask: Option<&ArrayD<i64>>,
        decoder_input_ids: Option<&ArrayD<i64>>,
        layer_states: Option<&Cache>,
    ) -> Result<LMModelOutput, RustBertError> {
        let calc_encoder_output = if encoder_hidden_states.is_none() {
            Some(
                self.encoder
                    .forward(input_ids, encoder_attention_mask, None, None, None)?
                    .last_hidden_state
                    .ok_or(RustBertError::ValueError(
                        "`last_hidden_state` not found in ONNX model outputs.".to_string(),
                    ))?,
            )
        } else {
            None
        };
        let encoder_hidden_states =
            encoder_hidden_states.unwrap_or_else(|| calc_encoder_output.as_ref().unwrap());

        let calc_encoder_attention_mask = if encoder_attention_mask.is_none() {
            Some(crate::common::tensor_ops::ones_i64(
                &encoder_hidden_states.shape()[..2],
            ))
        } else {
            None
        };
        let encoder_attention_mask =
            encoder_attention_mask.unwrap_or_else(|| calc_encoder_attention_mask.as_ref().unwrap());

        match (
            &self.decoder_without_past,
            &self.decoder_with_past,
            layer_states,
        ) {
            (Some(ref decoder_without_past), _, None)
            | (Some(ref decoder_without_past), _, Some(Cache::None)) => decoder_without_past
                .forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
                    Some(encoder_attention_mask),
                    None,
                    None,
                ),
            (_, Some(ref decoder_with_past), Some(Cache::ONNXCache(ref onnx_cache))) => {
                decoder_with_past.forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
                    Some(encoder_attention_mask),
                    None,
                    Some(onnx_cache),
                )
            }
            (Some(ref decoder_without_past), None, Some(Cache::ONNXCache(_))) => {
                decoder_without_past.forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
                    Some(encoder_attention_mask),
                    None,
                    None,
                )
            }
            (None, _, None) => Err(RustBertError::ValueError(
                "No decoder_without_cache loaded and no cache provided.".to_string(),
            )),
            (None, None, _) => Err(RustBertError::ValueError(
                "No decoder provided.".to_string(),
            )),
            (_, _, Some(cache)) => Err(RustBertError::ValueError(format!(
                "Invalid cache type provided, expected Cache::ONNXLayerCache, got {:?}.",
                cache
            ))),
        }
    }
}

#[cfg(feature = "libtorch")]
impl PrivateLanguageGenerator for ONNXConditionalGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }
    fn get_device(&self) -> tch::Device {
        tch::Device::Cpu
    }
    fn get_var_store_mut(&mut self) -> Result<&mut tch::nn::VarStore, RustBertError> {
        Err(RustBertError::ValueError(
            "No VarStore available for ONNX models".to_string(),
        ))
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
    fn get_forced_bos_token_id(&self) -> Option<i64> {
        self.forced_bos_token_id
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
    fn get_max_positions_embeddings(&self) -> Option<i64> {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&tch::Tensor>,
        layer_past: Cache,
        attention_mask: Option<&tch::Tensor>,
        _token_type_ids: Option<&tch::Tensor>,
        _position_ids: Option<&tch::Tensor>,
        _input_embeds: Option<&tch::Tensor>,
        encoder_outputs: Option<&tch::Tensor>,
        decoder_input_ids: Option<&tch::Tensor>,
        _train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let input_ids = input_ids.map(tensor_to_array_i64).transpose()?;
        let attention_mask = attention_mask.map(tensor_to_array_i64).transpose()?;
        let encoder_outputs = encoder_outputs.map(tensor_to_array_f32).transpose()?;
        let decoder_input_ids = decoder_input_ids.map(tensor_to_array_i64).transpose()?;
        self.forward(
            input_ids.as_ref(),
            attention_mask.as_ref(),
            encoder_outputs.as_ref(),
            None,
            decoder_input_ids.as_ref(),
            Some(&layer_past),
        )
    }

    fn encode(
        &self,
        input_ids: &tch::Tensor,
        attention_mask: Option<&tch::Tensor>,
    ) -> Option<tch::Tensor> {
        let input_ids = tensor_to_array_i64(input_ids).ok()?;
        let attention_mask = attention_mask.map(tensor_to_array_i64).transpose().ok()?;
        self.encoder
            .forward(Some(&input_ids), attention_mask.as_ref(), None, None, None)
            .ok()?
            .last_hidden_state
            .map(|array| array_to_tensor_f32(&array).expect("Error converting encoder output"))
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: tch::Tensor,
        encoder_outputs: Option<&'a tch::Tensor>,
        past: Cache,
        attention_mask: tch::Tensor,
    ) -> PreparedInput<'a> {
        match (past, self.use_past) {
            (Cache::ONNXCache(past), true) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::ONNXCache(past),
            },
            _ => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::None,
            },
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<tch::Tensor>,
        beam_indices: &tch::Tensor,
    ) -> Option<tch::Tensor> {
        let beam_indices = tensor_to_vec_i64(beam_indices).expect("Error converting beam indices");
        let beam_indices_tensor = tch::Tensor::from_slice(&beam_indices);
        let encoder_outputs =
            encoder_outputs.map(|value| value.index_select(0, &beam_indices_tensor));
        match past {
            Cache::ONNXCache(cached_decoder_state) => {
                for layer_past in cached_decoder_state.values.values_mut() {
                    *layer_past = crate::common::tensor_ops::gather_rows(layer_past, &beam_indices);
                }
            }
            Cache::None => {}
            _ => {
                panic!("Invalid cache for ONNX model");
            }
        }
        encoder_outputs
    }
}

#[cfg(feature = "libtorch")]
impl LanguageGenerator for ONNXConditionalGenerator {}

#[derive(Debug)]
/// Container used to store key-value cached states for efficient decoding.
pub struct ONNXLayerCache {
    pub values: HashMap<String, ArrayD<f32>>,
}

impl ONNXLayerCache {
    /// Helper function to create a cache layer from an ONNX model output.
    /// Assumes that the output names for cached keys and values contain `key` and `value` in their name, respectively.
    pub fn from_ort_output(
        ort_output: &SessionOutputs,
        key_value_names: &HashMap<String, usize>,
    ) -> Result<ONNXLayerCache, RustBertError> {
        Ok(ONNXLayerCache {
            values: crate::pipelines::onnx::conversion::key_values_from_outputs(
                ort_output,
                key_value_names,
            )?,
        })
    }
}
