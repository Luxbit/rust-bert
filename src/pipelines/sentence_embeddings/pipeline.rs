#[cfg(feature = "libtorch")]
use std::borrow::Borrow;
#[cfg(feature = "libtorch")]
use std::convert::{TryFrom, TryInto};

#[cfg(feature = "onnx")]
use crate::pipelines::onnx::ONNXEncoder;
use ndarray::Array2;
use rust_tokenizers::tokenizer::TruncationStrategy;
#[cfg(feature = "libtorch")]
use tch::{nn, Tensor};

#[cfg(feature = "libtorch")]
use crate::albert::AlbertForSentenceEmbeddings;
#[cfg(feature = "libtorch")]
use crate::bert::BertForSentenceEmbeddings;
#[cfg(feature = "libtorch")]
use crate::distilbert::DistilBertForSentenceEmbeddings;
#[cfg(feature = "libtorch")]
use crate::pipelines::common::ConfigOption;
use crate::pipelines::common::{ModelType, TokenizerOption};
#[cfg(feature = "libtorch")]
use crate::pipelines::sentence_embeddings::layers::Dense;
use crate::pipelines::sentence_embeddings::layers::{DenseConfig, Pooling, PoolingConfig};
#[cfg(feature = "libtorch")]
use crate::pipelines::sentence_embeddings::{AttentionHead, AttentionLayer, AttentionOutput};
use crate::pipelines::sentence_embeddings::{
    Embedding, SentenceEmbeddingsConfig, SentenceEmbeddingsModulesConfig,
    SentenceEmbeddingsSentenceBertConfig, SentenceEmbeddingsTokenizerConfig,
};
#[cfg(feature = "libtorch")]
use crate::roberta::RobertaForSentenceEmbeddings;
#[cfg(feature = "libtorch")]
use crate::t5::T5ForSentenceEmbeddings;
use crate::{Config, RustBertError};

/// # Abstraction that holds one particular sentence embeddings model, for any of the supported models
pub enum SentenceEmbeddingsOption {
    /// Bert for Sentence Embeddings
    #[cfg(feature = "libtorch")]
    Bert(BertForSentenceEmbeddings),
    /// DistilBert for Sentence Embeddings
    #[cfg(feature = "libtorch")]
    DistilBert(DistilBertForSentenceEmbeddings),
    /// Roberta for Sentence Embeddings
    #[cfg(feature = "libtorch")]
    Roberta(RobertaForSentenceEmbeddings),
    /// Albert for Sentence Embeddings
    #[cfg(feature = "libtorch")]
    Albert(AlbertForSentenceEmbeddings),
    /// T5 for Sentence Embeddings
    #[cfg(feature = "libtorch")]
    T5(T5ForSentenceEmbeddings),
    /// ONNX encoder for Sentence Embeddings. Pooling/dense/normalization modules
    /// are assumed to be part of the exported ONNX graph.
    #[cfg(feature = "onnx")]
    Onnx(ONNXEncoder),
}

impl SentenceEmbeddingsOption {
    /// Instantiate a new sentence embeddings transformer of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `transformer_type` - `ModelType` indicating the transformer model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. rust_model.ot)
    /// * `config` - A configuration (the transformer model type of the configuration must be compatible with the value for `transformer_type`)
    #[cfg(feature = "libtorch")]
    pub fn new<'p, P>(
        transformer_type: ModelType,
        p: P,
        config: &ConfigOption,
    ) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        use SentenceEmbeddingsOption::*;

        let option = match transformer_type {
            #[cfg(feature = "libtorch")]
            ModelType::Bert => Bert(BertForSentenceEmbeddings::new(p, &(config.try_into()?))),
            #[cfg(feature = "libtorch")]
            ModelType::DistilBert => DistilBert(DistilBertForSentenceEmbeddings::new(
                p,
                &(config.try_into()?),
            )),
            #[cfg(feature = "libtorch")]
            ModelType::Roberta => Roberta(RobertaForSentenceEmbeddings::new_with_optional_pooler(
                p,
                &(config.try_into()?),
                false,
            )),
            #[cfg(feature = "libtorch")]
            ModelType::Albert => Albert(AlbertForSentenceEmbeddings::new(p, &(config.try_into()?))),
            #[cfg(feature = "libtorch")]
            ModelType::T5 => T5(T5ForSentenceEmbeddings::new(p, &(config.try_into()?))),
            _ => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Unsupported transformer model {transformer_type:?} for Sentence Embeddings"
                )));
            }
        };

        Ok(option)
    }

    /// Interface method to forward() of the particular transformer models.
    #[cfg(feature = "libtorch")]
    pub fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        #[cfg(feature = "onnx")]
        if let Self::Onnx(encoder) = self {
            let ids = crate::common::tensor_conversion::tensor_to_array_i64(tokens_ids)?;
            let mask = crate::common::tensor_conversion::tensor_to_array_i64(tokens_masks)?;
            let hidden = encoder
                .forward(
                    Some(&ids.into_dyn()),
                    Some(&mask.into_dyn()),
                    None,
                    None,
                    None,
                )?
                .last_hidden_state
                .ok_or_else(|| {
                    RustBertError::ValueError(
                        "ONNX model did not return last_hidden_state".to_string(),
                    )
                })?;
            let logits = crate::common::tensor_conversion::array_to_tensor_f32(&hidden)?;
            return Ok((logits, None));
        }
        #[cfg(feature = "onnx")]
        if let Self::Onnx(encoder) = self {
            let ids = crate::common::tensor_conversion::tensor_to_array_i64(tokens_ids)?;
            let mask = crate::common::tensor_conversion::tensor_to_array_i64(tokens_masks)?;
            let hidden = encoder
                .forward(
                    Some(&ids.into_dyn()),
                    Some(&mask.into_dyn()),
                    None,
                    None,
                    None,
                )?
                .last_hidden_state
                .ok_or_else(|| {
                    RustBertError::ValueError(
                        "ONNX model did not return last_hidden_state".to_string(),
                    )
                })?;
            let logits = crate::common::tensor_conversion::array_to_tensor_f32(&hidden)?;
            return Ok((logits, None));
        }
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(encoder) => {
                let ids = crate::common::tensor_conversion::tensor_to_array_i64(tokens_ids)?;
                let mask = crate::common::tensor_conversion::tensor_to_array_i64(tokens_masks)?;
                let hidden = encoder
                    .forward(
                        Some(&ids.into_dyn()),
                        Some(&mask.into_dyn()),
                        None,
                        None,
                        None,
                    )?
                    .last_hidden_state
                    .ok_or_else(|| {
                        RustBertError::ValueError(
                            "ONNX model did not return last_hidden_state".to_string(),
                        )
                    })?;
                let logits = crate::common::tensor_conversion::array_to_tensor_f32(&hidden)?;
                Ok((logits, None))
            }
            #[cfg(feature = "libtorch")]
            Self::Bert(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            #[cfg(feature = "libtorch")]
            Self::DistilBert(transformer) => transformer
                .forward_t(Some(tokens_ids), Some(tokens_masks), None, false)
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            #[cfg(feature = "libtorch")]
            Self::Roberta(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            #[cfg(feature = "libtorch")]
            Self::Albert(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions.map(|attentions| {
                            attentions
                                .into_iter()
                                .map(|tensors| {
                                    let num_inner_groups = tensors.len() as f64;
                                    tensors.into_iter().sum::<Tensor>() / num_inner_groups
                                })
                                .collect()
                        }),
                    )
                }),
            #[cfg(feature = "libtorch")]
            Self::T5(transformer) => transformer.forward(tokens_ids, tokens_masks),
        }
    }
}

/// # SentenceEmbeddingsModel to perform sentence embeddings
///
/// It is made of the following blocks:
/// - `transformer`: Base transformer model
/// - `pooling`: Pooling layer
/// - `dense` _(optional)_: Linear (feed forward) layer
/// - `normalization` _(optional)_: Embeddings normalization
pub struct SentenceEmbeddingsModel {
    sentence_bert_config: SentenceEmbeddingsSentenceBertConfig,
    tokenizer: TokenizerOption,
    tokenizer_truncation_strategy: TruncationStrategy,
    #[cfg(feature = "libtorch")]
    var_store: nn::VarStore,
    transformer: SentenceEmbeddingsOption,
    #[cfg(feature = "libtorch")]
    transformer_config: Option<ConfigOption>,
    pooling_layer: Pooling,
    #[cfg(feature = "libtorch")]
    dense_layer: Option<Dense>,
    normalize_embeddings: bool,
    embeddings_dim: i64,
}

impl SentenceEmbeddingsModel {
    /// Build a new `SentenceEmbeddingsModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `SentenceEmbeddingsConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    pub fn new(config: SentenceEmbeddingsConfig) -> Result<Self, RustBertError> {
        let transformer_type = config.transformer_type;
        let tokenizer_vocab_resource = &config.tokenizer_vocab_resource;
        let tokenizer_merges_resource = &config.tokenizer_merges_resource;
        let tokenizer_config_resource = &config.tokenizer_config_resource;
        let sentence_bert_config_resource = &config.sentence_bert_config_resource;
        let tokenizer_config = SentenceEmbeddingsTokenizerConfig::from_file(
            tokenizer_config_resource.get_local_path()?,
        );
        let sentence_bert_config = SentenceEmbeddingsSentenceBertConfig::from_file(
            sentence_bert_config_resource.get_local_path()?,
        );

        let tokenizer = TokenizerOption::from_file(
            transformer_type,
            tokenizer_vocab_resource
                .get_local_path()?
                .to_string_lossy()
                .as_ref(),
            tokenizer_merges_resource
                .as_ref()
                .map(|resource| resource.get_local_path())
                .transpose()?
                .map(|path| path.to_string_lossy().into_owned())
                .as_deref(),
            tokenizer_config
                .do_lower_case
                .unwrap_or(sentence_bert_config.do_lower_case),
            tokenizer_config.strip_accents,
            tokenizer_config.add_prefix_space,
        )?;

        Self::new_with_tokenizer(config, tokenizer)
    }

    /// Build a new `ONNXCausalGenerator` from a `GenerateConfig` and `TokenizerOption`.
    ///
    /// A tokenizer must be provided by the user and can be customized to use non-default settings.
    ///
    /// # Arguments
    ///
    /// * `config` - `SentenceEmbeddingsConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for question answering.
    pub fn new_with_tokenizer(
        config: SentenceEmbeddingsConfig,
        tokenizer: TokenizerOption,
    ) -> Result<Self, RustBertError> {
        let SentenceEmbeddingsConfig {
            modules_config_resource,
            sentence_bert_config_resource,
            tokenizer_config_resource: _,
            tokenizer_vocab_resource: _,
            tokenizer_merges_resource: _,
            transformer_type,
            transformer_config_resource,
            transformer_weights_resource,
            pooling_config_resource,
            dense_config_resource,
            dense_weights_resource,
            device,
            #[cfg(feature = "libtorch")]
            kind,
            ..
        } = config;

        let modules =
            SentenceEmbeddingsModulesConfig::from_file(modules_config_resource.get_local_path()?)
                .validate()?;

        let sentence_bert_config = SentenceEmbeddingsSentenceBertConfig::from_file(
            sentence_bert_config_resource.get_local_path()?,
        );

        #[cfg_attr(not(feature = "libtorch"), allow(unused_variables))]
        let dense_out_features: Option<i64> = modules.dense_module().and_then(|_| {
            dense_config_resource
                .as_ref()
                .and_then(|resource| resource.get_local_path().ok())
                .map(|path| DenseConfig::from_file(path).out_features)
        });

        #[cfg(all(feature = "onnx", feature = "remote"))]
        let onnx_encoder = if transformer_type == ModelType::ONNX {
            Some(ONNXEncoder::new(
                transformer_weights_resource.get_local_path()?,
                &crate::pipelines::onnx::config::ONNXEnvironmentConfig::from_device(device),
            )?)
        } else {
            None
        };
        #[cfg(not(all(feature = "onnx", feature = "remote")))]
        let _onnx_encoder: Option<()> = {
            let _ = transformer_type;
            None
        };

        #[cfg(all(feature = "libtorch", feature = "onnx"))]
        let (transformer, transformer_config, var_store, dense_layer, torch_dense_out_features) =
            if onnx_encoder.is_some() {
                (
                    SentenceEmbeddingsOption::Onnx(onnx_encoder.unwrap()),
                    None,
                    nn::VarStore::new(tch::Device::Cpu),
                    None,
                    dense_out_features,
                )
            } else {
                let mut var_store = nn::VarStore::new(device.into());
                let transformer_config = ConfigOption::from_file(
                    transformer_type,
                    transformer_config_resource.get_local_path()?,
                );
                let transformer = SentenceEmbeddingsOption::new(
                    transformer_type,
                    var_store.root(),
                    &transformer_config,
                )?;
                crate::resources::load_weights(
                    &transformer_weights_resource,
                    &mut var_store,
                    kind,
                    device.into(),
                )?;

                let dense_layer = if modules.dense_module().is_some() {
                    let dense_config =
                        DenseConfig::from_file(dense_config_resource.unwrap().get_local_path()?);
                    Some(Dense::new(
                        dense_config,
                        dense_weights_resource.unwrap().get_local_path()?,
                        device.into(),
                    )?)
                } else {
                    None
                };
                (
                    transformer,
                    Some(transformer_config),
                    var_store,
                    dense_layer,
                    dense_out_features,
                )
            };
        #[cfg(all(feature = "libtorch", not(feature = "onnx")))]
        let (transformer, transformer_config, var_store, dense_layer, torch_dense_out_features) = {
            let mut var_store = nn::VarStore::new(device.into());
            let transformer_config = ConfigOption::from_file(
                transformer_type,
                transformer_config_resource.get_local_path()?,
            );
            let transformer = SentenceEmbeddingsOption::new(
                transformer_type,
                var_store.root(),
                &transformer_config,
            )?;
            crate::resources::load_weights(
                &transformer_weights_resource,
                &mut var_store,
                kind,
                device.into(),
            )?;

            let dense_layer = if modules.dense_module().is_some() {
                let dense_config =
                    DenseConfig::from_file(dense_config_resource.unwrap().get_local_path()?);
                Some(Dense::new(
                    dense_config,
                    dense_weights_resource.unwrap().get_local_path()?,
                    device.into(),
                )?)
            } else {
                None
            };
            (
                transformer,
                Some(transformer_config),
                var_store,
                dense_layer,
                dense_out_features,
            )
        };
        #[cfg(not(feature = "libtorch"))]
        let _ = (&transformer_config_resource, &dense_weights_resource);
        #[cfg(not(feature = "libtorch"))]
        let (transformer, _transformer_config, _dense_layer, _torch_dense_out_features) = {
            (
                SentenceEmbeddingsOption::Onnx(onnx_encoder.unwrap()),
                None::<()>,
                None::<()>,
                None::<i64>,
            )
        };

        // Setup pooling layer
        let pooling_config = PoolingConfig::from_file(pooling_config_resource.get_local_path()?);
        #[cfg_attr(not(feature = "libtorch"), allow(unused_mut))]
        let mut embeddings_dim = pooling_config.word_embedding_dimension;
        let pooling_layer = Pooling::new(pooling_config);

        #[cfg(all(feature = "libtorch", feature = "onnx"))]
        if let Some(out_features) = torch_dense_out_features {
            embeddings_dim = out_features;
        }
        #[cfg(all(feature = "libtorch", not(feature = "onnx")))]
        if let Some(out_features) = torch_dense_out_features {
            embeddings_dim = out_features;
        }

        let normalize_embeddings = modules.has_normalization();

        #[cfg(feature = "libtorch")]
        {
            Ok(Self {
                tokenizer,
                sentence_bert_config,
                tokenizer_truncation_strategy: TruncationStrategy::LongestFirst,
                var_store,
                transformer,
                transformer_config,
                pooling_layer,
                dense_layer,
                normalize_embeddings,
                embeddings_dim,
            })
        }
        #[cfg(not(feature = "libtorch"))]
        {
            Ok(Self {
                tokenizer,
                sentence_bert_config,
                tokenizer_truncation_strategy: TruncationStrategy::LongestFirst,
                transformer,
                pooling_layer,
                normalize_embeddings,
                embeddings_dim,
            })
        }
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }

    /// Sets the tokenizer's truncation strategy
    pub fn set_tokenizer_truncation(&mut self, truncation_strategy: TruncationStrategy) {
        self.tokenizer_truncation_strategy = truncation_strategy;
    }

    /// Return the embedding output dimension
    pub fn get_embedding_dim(&self) -> Result<i64, RustBertError> {
        Ok(self.embeddings_dim)
    }

    /// Tokenizes the inputs into padded token ids and attention masks (backend-neutral).
    pub fn tokenize_arrays<S>(&self, inputs: &[S]) -> (Array2<i64>, Array2<i64>)
    where
        S: AsRef<str> + Send + Sync,
    {
        let tokenized_input = self.tokenizer.encode_list(
            inputs,
            self.sentence_bert_config.max_seq_length,
            &self.tokenizer_truncation_strategy,
            0,
        );

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or(0);

        let pad_token_id = self.tokenizer.get_pad_id().unwrap_or(0);
        let mut tokens_ids = Array2::<i64>::zeros((tokenized_input.len(), max_len));
        let mut tokens_masks = Array2::<i64>::zeros((tokenized_input.len(), max_len));
        for (row, input) in tokenized_input.into_iter().enumerate() {
            let mut token_ids = input.token_ids;
            let padding = max_len - token_ids.len();
            tokens_masks
                .row_mut(row)
                .slice_mut(ndarray::s![padding..])
                .fill(1);
            token_ids.extend(vec![pad_token_id; padding]);
            tokens_ids
                .row_mut(row)
                .assign(&ndarray::Array1::from(token_ids));
        }
        (tokens_ids, tokens_masks)
    }

    /// Tokenizes the inputs
    #[cfg(feature = "libtorch")]
    pub fn tokenize<S>(&self, inputs: &[S]) -> SentenceEmbeddingsTokenizerOutput
    where
        S: AsRef<str> + Send + Sync,
    {
        let (tokens_ids, tokens_masks) = self.tokenize_arrays(inputs);
        let tokens_ids = tokens_ids
            .rows()
            .into_iter()
            .map(|row| Tensor::from_slice(&row.to_vec()))
            .collect::<Vec<_>>();
        let tokens_masks = tokens_masks
            .rows()
            .into_iter()
            .map(|row| Tensor::from_slice(&row.to_vec()))
            .collect::<Vec<_>>();

        SentenceEmbeddingsTokenizerOutput {
            tokens_ids,
            tokens_masks,
        }
    }

    /// Computes sentence embeddings, outputs `Tensor`.
    #[cfg(feature = "libtorch")]
    pub fn encode_as_tensor<S>(
        &self,
        inputs: &[S],
    ) -> Result<SentenceEmbeddingsModelOutput, RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        let SentenceEmbeddingsTokenizerOutput {
            tokens_ids,
            tokens_masks,
        } = self.tokenize(inputs);
        if tokens_ids.is_empty() {
            return Err(RustBertError::ValueError(
                "No n-gram found in the document. \
                Try allowing smaller n-gram sizes or relax stopword/forbidden characters criteria."
                    .to_string(),
            ));
        }
        let tokens_ids = Tensor::stack(&tokens_ids, 0).to(self.var_store.device());
        let tokens_masks = Tensor::stack(&tokens_masks, 0).to(self.var_store.device());

        let (tokens_embeddings, all_attentions) =
            tch::no_grad(|| self.transformer.forward(&tokens_ids, &tokens_masks))?;

        let mean_pool =
            tch::no_grad(|| self.pooling_layer.forward(tokens_embeddings, &tokens_masks));
        let maybe_linear = if let Some(dense_layer) = &self.dense_layer {
            tch::no_grad(|| dense_layer.forward(&mean_pool))
        } else {
            mean_pool
        };
        let maybe_normalized = if self.normalize_embeddings {
            let norm = &maybe_linear
                .norm_scalaropt_dim(2, [1], true)
                .clamp_min(1e-12)
                .expand_as(&maybe_linear);
            maybe_linear / norm
        } else {
            maybe_linear
        };

        Ok(SentenceEmbeddingsModelOutput {
            embeddings: maybe_normalized,
            all_attentions,
        })
    }

    /// Computes sentence embeddings.
    pub fn encode<S>(&self, inputs: &[S]) -> Result<Vec<Embedding>, RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        let embeddings = self.encode_arrays(inputs)?;
        Ok(embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect())
    }

    /// Computes sentence embeddings as an (n, dim) array, for both backends.
    pub fn encode_arrays<S>(&self, inputs: &[S]) -> Result<Array2<f32>, RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
        let (tokens_ids, tokens_masks) = self.tokenize_arrays(inputs);
        if tokens_ids.nrows() == 0 {
            return Err(RustBertError::ValueError(
                "No n-gram found in the document. \
                Try allowing smaller n-gram sizes or relax stopword/forbidden characters criteria."
                    .to_string(),
            ));
        }

        #[cfg(feature = "onnx")]
        #[cfg_attr(not(feature = "libtorch"), allow(irrefutable_let_patterns))]
        if let SentenceEmbeddingsOption::Onnx(encoder) = &self.transformer {
            let tokens_masks_ref = tokens_masks.clone();
            let output = encoder
                .forward(
                    Some(&tokens_ids.into_dyn()),
                    Some(&tokens_masks.into_dyn()),
                    None,
                    None,
                    None,
                )?
                .last_hidden_state
                .ok_or_else(|| {
                    RustBertError::ValueError(
                        "ONNX model did not return last_hidden_state".to_string(),
                    )
                })?;
            // (batch, seq, hidden) -> Array3
            let token_embeddings = output.into_dimensionality::<ndarray::Ix3>().map_err(|_| {
                RustBertError::ValueError("Unexpected ONNX output rank".to_string())
            })?;
            let mut pooled = self
                .pooling_layer
                .forward_array(&token_embeddings, &tokens_masks_ref);
            if self.normalize_embeddings {
                pooled = Self::normalize_rows(&pooled);
            }
            return Ok(pooled);
        }

        #[cfg(feature = "libtorch")]
        {
            let SentenceEmbeddingsModelOutput { embeddings, .. } = self.encode_as_tensor(inputs)?;
            let embeddings = crate::common::tensor_conversion::tensor_to_array_f32(&embeddings)?;
            let embeddings = embeddings
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RustBertError::ValueError("Unexpected embeddings rank".to_string()))?;
            Ok(embeddings)
        }
        #[cfg(not(feature = "libtorch"))]
        {
            unreachable!("No inference backend available")
        }
    }

    /// L2-normalizes each row of the array.
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    fn normalize_rows(array: &Array2<f32>) -> Array2<f32> {
        let mut output = array.clone();
        for mut row in output.rows_mut().into_iter() {
            let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in row.iter_mut() {
                *value /= norm;
            }
        }
        output
    }

    #[cfg(feature = "libtorch")]
    fn nb_layers(&self) -> usize {
        use SentenceEmbeddingsOption::*;
        match (&self.transformer, &self.transformer_config) {
            #[cfg(feature = "onnx")]
            (SentenceEmbeddingsOption::Onnx(_), _) => 0,
            #[cfg(feature = "libtorch")]
            (Bert(_), Some(ConfigOption::Bert(conf))) => conf.num_hidden_layers as usize,
            #[cfg(feature = "libtorch")]
            (Bert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (DistilBert(_), Some(ConfigOption::DistilBert(conf))) => conf.n_layers as usize,
            #[cfg(feature = "libtorch")]
            (DistilBert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (Roberta(_), Some(ConfigOption::Bert(conf))) => conf.num_hidden_layers as usize,
            #[cfg(feature = "libtorch")]
            (Roberta(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (Albert(_), Some(ConfigOption::Albert(conf))) => conf.num_hidden_layers as usize,
            #[cfg(feature = "libtorch")]
            (Albert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (T5(_), Some(ConfigOption::T5(conf))) => conf.num_layers as usize,
            #[cfg(feature = "libtorch")]
            (T5(_), _) => unreachable!(),
        }
    }

    #[cfg(feature = "libtorch")]
    fn nb_heads(&self) -> usize {
        use SentenceEmbeddingsOption::*;
        match (&self.transformer, &self.transformer_config) {
            #[cfg(feature = "onnx")]
            (SentenceEmbeddingsOption::Onnx(_), _) => 0,
            #[cfg(feature = "libtorch")]
            (Bert(_), Some(ConfigOption::Bert(conf))) => conf.num_attention_heads as usize,
            #[cfg(feature = "libtorch")]
            (Bert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (DistilBert(_), Some(ConfigOption::DistilBert(conf))) => conf.n_heads as usize,
            #[cfg(feature = "libtorch")]
            (DistilBert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (Roberta(_), Some(ConfigOption::Roberta(conf))) => conf.num_attention_heads as usize,
            #[cfg(feature = "libtorch")]
            (Roberta(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (Albert(_), Some(ConfigOption::Albert(conf))) => conf.num_attention_heads as usize,
            #[cfg(feature = "libtorch")]
            (Albert(_), _) => unreachable!(),
            #[cfg(feature = "libtorch")]
            (T5(_), Some(ConfigOption::T5(conf))) => conf.num_heads as usize,
            #[cfg(feature = "libtorch")]
            (T5(_), _) => unreachable!(),
        }
    }

    /// Computes sentence embeddings, also outputs `AttentionOutput`s.
    #[cfg(feature = "libtorch")]
    pub fn encode_with_attention<S>(
        &self,
        inputs: &[S],
    ) -> Result<(Vec<Embedding>, Vec<AttentionOutput>), RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        let SentenceEmbeddingsModelOutput {
            embeddings,
            all_attentions,
        } = self.encode_as_tensor(inputs)?;

        let embeddings = Vec::try_from(embeddings)?;
        let all_attentions = all_attentions.ok_or_else(|| {
            RustBertError::InvalidConfigurationError("No attention outputted".into())
        })?;

        let attention_outputs = (0..inputs.len() as i64)
            .map(|i| {
                let mut attention_output = AttentionOutput::with_capacity(self.nb_layers());
                for layer in all_attentions.iter() {
                    let mut attention_layer = AttentionLayer::with_capacity(self.nb_heads());
                    for head in 0..self.nb_heads() {
                        let attention_slice = layer
                            .slice(0, i, i + 1, 1)
                            .slice(1, head as i64, head as i64 + 1, 1)
                            .squeeze();
                        let attention_head = AttentionHead::try_from(attention_slice).unwrap();
                        attention_layer.push(attention_head);
                    }
                    attention_output.push(attention_layer);
                }
                attention_output
            })
            .collect::<Vec<AttentionOutput>>();

        Ok((embeddings, attention_outputs))
    }
}

/// Container for the SentenceEmbeddings tokenizer output.
#[cfg(feature = "libtorch")]
pub struct SentenceEmbeddingsTokenizerOutput {
    pub tokens_ids: Vec<Tensor>,
    pub tokens_masks: Vec<Tensor>,
}

/// Container for the SentenceEmbeddings model output.
#[cfg(feature = "libtorch")]
pub struct SentenceEmbeddingsModelOutput {
    pub embeddings: Tensor,
    pub all_attentions: Option<Vec<Tensor>>,
}
