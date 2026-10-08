// Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors.
// Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

//! # Natural Language Generation utilities
//! Set of text generation utilities, serving as a basis for TextGenerationModel, SummarizationModels and TranslationModels.
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! Supports batch generation of sentences from several prompts. Sequences will be left-padded with the model's padding token if present, the unknown token otherwise.
//! This may impact the results and it is recommended to submit prompts of similar length for best results.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::gpt2::GPT2Generator;
//! use rust_bert::pipelines::generation_utils::{
//!     GenerateConfig, GenerateOptions, LanguageGenerator,
//! };
//!
//! let generate_config = GenerateConfig {
//!     do_sample: true,
//!     num_beams: 5,
//!     temperature: 1.1,
//!     num_return_sequences: 3,
//!     ..Default::default()
//! };
//! let mut gpt2_generator = GPT2Generator::new(generate_config)?;
//!
//! let input_context = "The dog";
//! let second_input_context = "The cat was";
//!
//! let generate_options = GenerateOptions {
//!     min_length: Some(32),
//!     max_length: Some(128),
//!     output_scores: true,
//!     ..Default::default()
//! };
//!
//! let output = gpt2_generator.generate(
//!     Some(&[input_context, second_input_context]),
//!     Some(generate_options),
//! );
//! # Ok(())
//! # }
//! ```
//!
//! Example output: \
//! ```no_run
//! # let output =
//! [
//!     "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year",
//!     "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me",
//!     "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's",
//!     "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,",
//!     "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said",
//!     "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
//! ]
//! # ;
//! ```

//! # Common logic for text generation
//! Shared logic for text generation, including beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! Implementations of the `LanguageGenerator` trait are available for the following models:
//! - GPT2
//! - GPT
//! - BART
//! - T5
//! - LongT5
//! - MBart
//! - M2M100
//! - NLLB
//! - Reformer
//! - XLNet
//! - GPT-Neo
//! - GPT-J
//! - Pegasus
//! - ProphetNet
//! - ONNX models (causal and conditional generators)
//!
//! The generation loop operates on `ndarray` arrays end-to-end, making it available for both the
//! LibTorch and ONNX (ort) backends. Model implementations convert at the backend boundary: the
//! key/value caches carried through the loop stay opaque and are only manipulated by the
//! generators themselves (via `reorder_cache`).

use crate::pipelines::generation_utils::private_generation_utils::{
    InternalGenerateOptions, PrivateLanguageGenerator,
};

use self::ordered_float::OrderedFloat;
use crate::common::resources::ResourceProvider;
use crate::pipelines::common::{ModelResource, ModelType, TokenizerOption};
use crate::Device;

extern crate ordered_float;
#[cfg(feature = "onnx")]
use crate::pipelines::onnx::ONNXLayerCache;
use crate::RustBertError;
#[cfg(feature = "libtorch")]
use crate::{
    bart::LayerState as BartLayerState, gpt_j::LayerState as GPTJLayerState,
    gpt_neo::LayerState as GPTNeoLayerState, prophetnet::LayerState as ProphetNetLayerState,
    reformer::LayerState as ReformerLayerState, t5::LayerState as T5LayerState,
    xlnet::LayerState as XLNetLayerState,
};
#[cfg(all(feature = "remote", feature = "libtorch"))]
use crate::{
    gpt2::{Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources},
    resources::RemoteResource,
};
use ndarray::{Array2, ArrayD, Ix2};
#[cfg(feature = "libtorch")]
use tch::Tensor;

/// # Configuration for text generation
pub struct GenerateConfig {
    /// Model type used for generation
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained GPT2 model)
    pub model_resource: ModelResource,
    /// Config resource (default: pretrained GPT2 model)
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource (default: pretrained GPT2 model)
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource (default: pretrained GPT2 model)
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 20)
    pub max_length: Option<i64>,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding (default: true)
    pub do_sample: bool,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated (default: false)
    pub early_stopping: bool,
    /// Number of beams for beam search (default: 5)
    pub num_beams: i64,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance (default: 1.0)
    pub temperature: f64,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature (default: 0)
    pub top_k: i64,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p (default: 0.9)
    pub top_p: f64,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated. (default: 1.0)
    pub repetition_penalty: f64,
    /// Exponential penalty based on the length of the hypotheses generated (default: 1.0)
    pub length_penalty: f64,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature (default: 3)
    pub no_repeat_ngram_size: i64,
    /// Number of sequences to return for each prompt text (default: 1)
    pub num_return_sequences: i64,
    /// Number of beam groups for diverse beam generation. If provided and higher than 1, will split the beams into beam subgroups leading to more diverse generation.
    pub num_beam_groups: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups (default: 5.5)
    pub diversity_penalty: Option<f64>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Model weights precision (LibTorch backend only). If not provided, will default to full precision on CPU, or the loaded weights precision otherwise
    #[cfg(feature = "libtorch")]
    pub kind: Option<tch::Kind>,
}

#[cfg(all(feature = "remote", feature = "libtorch"))]
impl Default for GenerateConfig {
    fn default() -> GenerateConfig {
        GenerateConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
                Gpt2ModelResources::GPT2,
            ))),
            config_resource: Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2)),
            vocab_resource: Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2)),
            merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                Gpt2MergesResources::GPT2,
            ))),
            min_length: 0,
            max_length: Some(56),
            do_sample: true,
            early_stopping: true,
            num_beams: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 3,
            num_return_sequences: 1,
            num_beam_groups: None,
            diversity_penalty: None,
            device: crate::Device::cuda_if_available(),
            kind: None,
        }
    }
}

impl GenerateConfig {
    #[cfg_attr(not(feature = "libtorch"), allow(dead_code))]
    pub(crate) fn validate(&self) {
        assert!(self.temperature > 0f64, "temperature must positive");
        assert!(
            (self.top_p >= 0f64) & (self.top_p <= 1f64),
            "top_p must be 0 and 1"
        );
        assert!(
            self.repetition_penalty >= 1f64,
            "repetition_penalty must be greater than 1"
        );
        assert!(
            self.length_penalty > 0f64,
            "length_penalty must be strictly greater than 0"
        );
        assert!(
            self.num_return_sequences > 0i64,
            "num_return_sequences must be strictly greater than 0"
        );
        assert!(
            self.num_beams > 0i64,
            "num_beams must be strictly greater than 0"
        );

        if !self.do_sample {
            if self.num_beams == 1 {
                assert_eq!(
                    self.num_return_sequences, 1,
                    "num_return_sequences must be set to 1 for greedy decoding"
                )
            } else {
                assert!(
                    self.num_beams >= self.num_return_sequences,
                    "num_return_sequences must be lower than the number of beams"
                )
            }
        }
        if let Some(num_beam_groups_value) = self.num_beam_groups {
            if num_beam_groups_value > 1 {
                assert_eq!(
                    self.num_beams % num_beam_groups_value,
                    0,
                    "num_beam_groups must be a multiple of num_beam_groups"
                )
            }
        }
    }
}

#[derive(Debug)]
pub enum Cache {
    #[cfg(feature = "libtorch")]
    GPT2Cache(Option<Vec<tch::Tensor>>),
    #[cfg(feature = "libtorch")]
    BARTCache(Option<Vec<(Option<BartLayerState>, Option<BartLayerState>)>>),
    #[cfg(feature = "libtorch")]
    T5Cache(Option<Vec<(Option<T5LayerState>, Option<T5LayerState>)>>),
    #[cfg(feature = "libtorch")]
    LongT5Cache(Option<Vec<(Option<T5LayerState>, Option<T5LayerState>)>>),
    #[cfg(feature = "libtorch")]
    XLNetCache(Option<Vec<Option<XLNetLayerState>>>),
    #[cfg(feature = "libtorch")]
    ReformerCache(Option<Vec<Option<ReformerLayerState>>>),
    #[cfg(feature = "libtorch")]
    ProphetNetCache(Option<Vec<(Option<ProphetNetLayerState>, Option<ProphetNetLayerState>)>>),
    #[cfg(feature = "libtorch")]
    GPTNeoCache(Option<Vec<Option<GPTNeoLayerState>>>),
    #[cfg(feature = "libtorch")]
    GPTJCache(Option<Vec<Option<GPTJLayerState>>>),
    #[cfg(feature = "onnx")]
    ONNXCache(ONNXLayerCache),
    None,
}

/// Converts an optional 2-D integer array to an optional tensor (LibTorch only).
#[cfg(feature = "libtorch")]
pub(crate) fn option_array2_to_tensor(array: Option<&ndarray::Array2<i64>>) -> Option<Tensor> {
    array.map(|array| {
        crate::common::tensor_conversion::array_to_tensor_i64(&array.clone().into_dyn())
            .expect("Error converting array input to tensor")
    })
}

/// Converts an optional array to an optional tensor (LibTorch only).
#[cfg(feature = "libtorch")]
pub(crate) fn option_array_to_tensor_f32(array: Option<&ArrayD<f32>>) -> Option<Tensor> {
    array.map(|array| {
        crate::common::tensor_conversion::array_to_tensor_f32(array)
            .expect("Error converting array input to tensor")
    })
}

/// Append a column of values to the right of a 2-D array.
pub(crate) fn append_column(array: &Array2<i64>, column: &[i64]) -> Array2<i64> {
    let (rows, cols) = (array.nrows(), array.ncols());
    let mut out = Array2::<i64>::zeros((rows, cols + 1));
    out.slice_mut(ndarray::s![.., ..cols]).assign(array);
    for (row, &value) in column.iter().enumerate() {
        out[[row, cols]] = value;
    }
    out
}

/// Extract the last column of a 2-D integer array as a new 2-D array (n x 1).
#[cfg_attr(not(feature = "libtorch"), allow(dead_code))]
pub(crate) fn last_column(array: &ndarray::Array2<i64>) -> ndarray::Array2<i64> {
    ndarray::Array2::from_shape_vec((array.nrows(), 1), array.column(array.ncols() - 1).to_vec())
        .unwrap()
}

/// Index-select rows of a 2-D integer array.
pub(crate) fn index_select_rows_i64(array: &Array2<i64>, indices: &[i64]) -> Array2<i64> {
    crate::common::tensor_ops::gather_rows(&array.clone().into_dyn(), indices)
        .into_dimensionality::<Ix2>()
        .expect("Row selection should return a 2-D array")
}

/// Weighted sampling without replacement (Efraimidis-Spirakis). Returns the indices of the
/// sampled items, ordered by decreasing weight.
pub(crate) fn weighted_sample_without_replacement(
    weights: &[f32],
    num_samples: usize,
) -> Vec<usize> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut keys: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .filter(|(_, &weight)| weight > 0f32)
        .map(|(index, &weight)| {
            let u: f64 = rng.random::<f64>().clamp(f64::EPSILON, 1.0);
            ((u.ln() / weight as f64), index)
        })
        .collect();
    keys.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keys.truncate(num_samples);
    keys.into_iter().map(|(_, index)| index).collect()
}

pub(crate) mod private_generation_utils {
    use rust_tokenizers::TokenIdsWithOffsets;
    use std::cmp::{max, min};
    use std::collections::HashMap;

    use rust_tokenizers::tokenizer::{truncate_sequences, TruncationStrategy};
    #[cfg(feature = "libtorch")]
    use tch::nn;

    use crate::pipelines::common::TokenizerOption;
    use crate::pipelines::generation_utils::{
        append_column, index_select_rows_i64, weighted_sample_without_replacement, BeamHypotheses,
        Cache, GenerateConfig, GeneratedLogits, PrefixAllowedFunction,
    };

    use super::ordered_float::OrderedFloat;
    use crate::RustBertError;
    use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};

    const NEG_INF: f32 = f32::NEG_INFINITY;

    /// Runs the provided closure under a LibTorch no-grad context (no-op for the ONNX backend).
    #[cfg(feature = "libtorch")]
    pub(crate) fn no_grad<T>(f: impl FnOnce() -> T) -> T {
        tch::no_grad(f)
    }

    #[cfg(not(feature = "libtorch"))]
    pub(crate) fn no_grad<T>(f: impl FnOnce() -> T) -> T {
        f()
    }

    pub struct InternalGenerateOptions<'a> {
        pub min_length: i64,
        pub max_length: Option<i64>,
        pub do_sample: bool,
        pub temperature: f64,
        pub top_k: i64,
        pub top_p: f64,
        pub repetition_penalty: f64,
        pub no_repeat_ngram_size: i64,
        pub pad_token_id: Option<i64>,
        pub eos_token_ids: Option<Vec<i64>>,
        pub num_return_sequences: i64,
        pub early_stopping: bool,
        pub num_beams: i64,
        pub length_penalty: f64,
        pub num_beam_groups: Option<i64>,
        pub diversity_penalty: Option<f64>,
        pub forced_bos_token_id: Option<i64>,
        pub bad_word_ids: Option<&'a Vec<Vec<i64>>>,
    }

    pub struct PreparedInput<'a> {
        pub prepared_input: Option<Array2<i64>>,
        pub prepared_attention_mask: Option<Array2<i64>>,
        pub prepared_encoder_output: Option<&'a ArrayD<f32>>,
        pub prepared_decoder_input: Option<Array2<i64>>,
        pub prepared_position_ids: Option<Array2<i64>>,
        pub prepared_past: Cache,
    }

    pub struct GeneratedOutputWithScores {
        pub indices: Array2<i64>,
        pub scores: Option<Vec<f64>>,
        pub token_scores: Option<Vec<Vec<f64>>>,
    }

    pub trait PrivateLanguageGenerator {
        fn _get_tokenizer(&self) -> &TokenizerOption;
        fn get_device(&self) -> crate::Device;
        #[cfg(feature = "libtorch")]
        fn get_var_store_mut(&mut self) -> Result<&mut nn::VarStore, RustBertError>;
        fn _get_tokenizer_mut(&mut self) -> &mut TokenizerOption;
        fn get_config(&self) -> &GenerateConfig;
        fn get_bos_id(&self) -> Option<i64>;
        fn get_eos_ids(&self) -> Option<&Vec<i64>>;
        fn get_forced_bos_token_id(&self) -> Option<i64> {
            None
        }
        fn get_forced_eos_token_id(&self) -> Option<i64> {
            None
        }
        fn get_pad_id(&self) -> Option<i64>;
        fn is_encoder_decoder(&self) -> bool;
        fn get_vocab_size(&self) -> i64;
        fn get_decoder_start_id(&self) -> Option<i64>;
        fn get_max_positions_embeddings(&self) -> Option<i64>;

        fn forward_t(
            &self,
            input_ids: Option<&Array2<i64>>,
            layer_past: Cache,
            attention_mask: Option<&Array2<i64>>,
            token_type_ids: Option<&Array2<i64>>,
            position_ids: Option<&Array2<i64>>,
            input_embeds: Option<&ArrayD<f32>>,
            encoder_outputs: Option<&ArrayD<f32>>,
            decoder_input_ids: Option<&Array2<i64>>,
            train: bool,
        ) -> Result<GeneratedLogits, RustBertError>;

        fn prepare_scores_for_generation(
            &self,
            scores: &mut Array2<f32>,
            current_length: i64,
            max_length: Option<i64>,
            forced_bos_token_id: Option<i64>,
        ) {
            if current_length == 1 {
                if let Some(forced_bos_token_id) =
                    forced_bos_token_id.or(self.get_forced_bos_token_id())
                {
                    force_token_id_generation(
                        scores,
                        &[forced_bos_token_id],
                        self.get_vocab_size(),
                    );
                }
            } else if let Some(max_length) = max_length {
                if let Some(forced_eos_token_id) = self.get_forced_eos_token_id() {
                    if current_length == max_length - 1 {
                        force_token_id_generation(
                            scores,
                            &[forced_eos_token_id],
                            self.get_vocab_size(),
                        );
                    }
                }
            }
        }

        fn encode(
            &self,
            _input_ids: &Array2<i64>,
            _attention_mask: Option<&Array2<i64>>,
        ) -> Option<ArrayD<f32>> {
            None
        }

        fn prepare_inputs_for_generation<'a>(
            &self,
            input_ids: Array2<i64>,
            _encoder_outputs: Option<&'a ArrayD<f32>>,
            past: Cache,
            attention_mask: Array2<i64>,
        ) -> PreparedInput<'a> {
            PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: None,
                prepared_past: past,
            }
        }

        fn encode_prompt_text<S>(
            &self,
            prompt_text: &[S],
            max_len: Option<i64>,
            pad_token_id: Option<i64>,
        ) -> Array2<i64>
        where
            S: AsRef<str> + Send + Sync,
        {
            let token_ids = if self.is_encoder_decoder() {
                let tokens = self._get_tokenizer().encode_list(
                    prompt_text,
                    max_len
                        .map(|max_len| max_len as usize)
                        .unwrap_or(usize::MAX),
                    &TruncationStrategy::LongestFirst,
                    0,
                );
                tokens
                    .into_iter()
                    .map(|tokenized_input| tokenized_input.token_ids)
                    .collect::<Vec<Vec<i64>>>()
            } else {
                // Special tokens (e.g. BOS) are not added at the end of the prompt for causal generation
                let tokens = self._get_tokenizer().tokenize_list(prompt_text);
                let token_ids = tokens
                    .into_iter()
                    .map(|prompt_tokens| {
                        self._get_tokenizer().convert_tokens_to_ids(&prompt_tokens)
                    })
                    .collect::<Vec<Vec<i64>>>();

                let num_truncated_tokens = token_ids
                    .iter()
                    .map(|token_ids| {
                        max_len
                            .map(|max_len| {
                                if token_ids.len() > max_len as usize {
                                    token_ids.len() - max_len as usize
                                } else {
                                    0
                                }
                            })
                            .unwrap_or(0)
                    })
                    .collect::<Vec<usize>>();

                token_ids
                    .into_iter()
                    .zip(num_truncated_tokens)
                    .map(|(tokens, num_truncated_tokens)| {
                        truncate_sequences(
                            TokenIdsWithOffsets {
                                ids: tokens,
                                offsets: vec![],
                                reference_offsets: vec![],
                                masks: vec![],
                            },
                            None,
                            num_truncated_tokens,
                            &TruncationStrategy::LongestFirst,
                            0,
                        )
                        .unwrap()
                        .0
                        .ids
                    })
                    .collect::<Vec<Vec<i64>>>()
            };

            let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

            let pad_token = match pad_token_id {
                Some(value) => value,
                None => self._get_tokenizer().get_unk_id(),
            };

            let rows = token_ids.len();
            let mut output = Array2::<i64>::zeros((rows, max_len));
            for (row, mut input) in token_ids.into_iter().enumerate() {
                let mut temp = vec![pad_token; max_len - input.len()];
                if self.is_encoder_decoder() {
                    input.extend(temp);
                } else {
                    // Pad left for causal generation
                    temp.extend(input);
                    input = temp;
                }
                output.row_mut(row).assign(&Array1::from(input));
            }
            output
        }

        fn enforce_repetition_penalty(
            &self,
            next_token_logits: &mut Array2<f32>,
            _batch_size: i64,
            _num_beams: i64,
            prev_output_tokens: &Array2<i64>,
            repetition_penalty: f64,
        ) {
            for row in 0..next_token_logits.nrows() {
                if row >= prev_output_tokens.nrows() {
                    break;
                }
                for &token in prev_output_tokens.row(row) {
                    let updated_value = next_token_logits[[row, token as usize]] as f64;
                    let penalized = if updated_value < 0f64 {
                        updated_value * repetition_penalty
                    } else {
                        updated_value / repetition_penalty
                    };
                    next_token_logits[[row, token as usize]] = penalized as f32;
                }
            }
        }

        fn get_banned_tokens(
            &self,
            input_ids: &Array2<i64>,
            no_repeat_ngram_size: i64,
            cur_len: i64,
        ) -> Vec<Vec<i64>> {
            //        Ported from hugging face's transformers and fairseq (https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py)
            if cur_len + 1 < no_repeat_ngram_size {
                vec![vec![]]
            } else {
                let num_hypothesis = input_ids.nrows();
                let mut banned_tokens: Vec<Vec<i64>> = Vec::with_capacity(num_hypothesis);
                for hypothesis_index in 0..num_hypothesis {
                    let hypothesis_input_ids = input_ids.row(hypothesis_index).to_vec();
                    let mut generated_ngram: HashMap<Vec<i64>, Vec<i64>> = HashMap::new();
                    let input: Vec<i64> = (0..hypothesis_input_ids.len() as i64).collect();
                    let query = &hypothesis_input_ids
                        [cur_len as usize + 1 - no_repeat_ngram_size as usize..]
                        .to_vec();
                    for ngram in input
                        .windows(no_repeat_ngram_size as usize)
                        .map(|win| (*win.first().unwrap(), *win.last().unwrap()))
                    {
                        let ngram = &hypothesis_input_ids[ngram.0 as usize..ngram.1 as usize + 1];
                        let key = ngram[..no_repeat_ngram_size as usize - 1].to_vec();
                        let value = *ngram.last().unwrap();
                        generated_ngram
                            .entry(key)
                            .or_insert_with(|| vec![value])
                            .push(value);
                    }
                    let hypothesis_banned_tokens = match generated_ngram.get(query) {
                        Some(banned_tokens) => banned_tokens.clone(),
                        None => vec![],
                    };
                    banned_tokens.push(hypothesis_banned_tokens);
                }
                banned_tokens
            }
        }

        fn top_k_top_p_filtering(
            &self,
            logits: &mut Array2<f32>,
            top_k: i64,
            top_p: f64,
            min_tokens_to_keep: i64,
        ) {
            //        Nucleus and top-k filtering introduced by Holtzman et al. (http://arxiv.org/abs/1904.09751)
            //        Ported from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            let vocab_size = logits.ncols() as i64;
            if top_k > 0 {
                let keep = min(max(top_k, min_tokens_to_keep), vocab_size) as usize;
                for mut row in logits.rows_mut().into_iter() {
                    let mut order: Vec<usize> = (0..row.len()).collect();
                    order.sort_unstable_by(|&a, &b| {
                        row[b]
                            .partial_cmp(&row[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for &remove_index in &order[keep..] {
                        row[remove_index] = NEG_INF;
                    }
                }
            }
            if top_p < 1f64 {
                let mut sorted_logits_rows: Vec<Vec<f32>> = Vec::with_capacity(logits.nrows());
                let mut sorted_indices_rows: Vec<Vec<usize>> = Vec::with_capacity(logits.nrows());
                for row in logits.rows().into_iter() {
                    let mut order: Vec<usize> = (0..row.len()).collect();
                    order.sort_unstable_by(|&a, &b| {
                        row[b]
                            .partial_cmp(&row[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let sorted: Vec<f32> = order.iter().map(|&index| row[index]).collect();
                    sorted_logits_rows.push(sorted);
                    sorted_indices_rows.push(order);
                }
                // Remove tokens with cumulative probability above the threshold
                let mut remove_flags: Vec<Vec<bool>> = Vec::with_capacity(sorted_logits_rows.len());
                for (row_index, sorted_row) in sorted_logits_rows.iter().enumerate() {
                    let total: f32 = sorted_row.iter().sum();
                    let mut cumulative = 0f32;
                    let mut flags = vec![false; sorted_row.len()];
                    for (position, &value) in sorted_row.iter().enumerate() {
                        if position > 0 && (cumulative / total) > top_p as f32 {
                            flags[position] = true;
                        }
                        cumulative += value;
                    }
                    if min_tokens_to_keep > 1 {
                        for flag in flags.iter_mut().take(min_tokens_to_keep as usize + 1) {
                            *flag = false;
                        }
                    }
                    let _ = row_index;
                    remove_flags.push(flags);
                }
                // Scatter back to the original token order
                for (row_index, order) in sorted_indices_rows.iter().enumerate() {
                    for (position, &token_index) in order.iter().enumerate() {
                        if remove_flags[row_index][position] {
                            logits[[row_index, token_index]] = NEG_INF;
                        }
                    }
                }
            }
        }

        fn run_hamming_diversity_penalty(
            &self,
            scores: &mut Array2<f32>,
            current_tokens: &Array1<i64>,
            diversity_penalty: f32,
            num_beams: i64,
            batch_size: i64,
            group_size: i64,
            group_start_index: i64,
        ) {
            if group_start_index > 0 {
                let vocab_size = scores.ncols();
                for batch_index in 0..batch_size {
                    let mut counts = vec![0i64; vocab_size];
                    for beam_index in
                        (batch_index * num_beams)..(batch_index * num_beams + group_start_index)
                    {
                        let token = current_tokens[beam_index as usize];
                        if token >= 0 && (token as usize) < vocab_size {
                            counts[token as usize] += 1;
                        }
                    }
                    for beam_index in (batch_index * group_size)..((batch_index + 1) * group_size) {
                        for (token, &count) in counts.iter().enumerate() {
                            if count > 0 {
                                scores[[beam_index as usize, token]] -=
                                    count as f32 * diversity_penalty;
                            }
                        }
                    }
                }
            }
        }

        fn apply_prefix_allowed_tokens_function(
            &self,
            prefix_allowed_tokens_fn: &dyn Fn(i64, &[i64]) -> Vec<i64>,
            num_beams: i64,
            input_ids: &Array2<i64>,
            scores: &mut Array2<f32>,
        ) {
            for idx in 0..scores.nrows() {
                let batch_id = (idx as i64) / num_beams;
                let row: Vec<i64> = input_ids.row(min(idx, input_ids.nrows() - 1)).to_vec();
                let allowed_tokens = prefix_allowed_tokens_fn(batch_id, &row);
                let mut mask = vec![get_positive_infinity_f32(); scores.ncols()];
                for token in allowed_tokens {
                    if (token as usize) < mask.len() {
                        mask[token as usize] = 0f32;
                    }
                }
                for (token, &penalty) in mask.iter().enumerate() {
                    scores[[idx, token]] -= penalty;
                }
            }
        }

        fn split_bad_word_ids<'a>(
            &self,
            bad_word_ids: Option<&'a Vec<Vec<i64>>>,
        ) -> (Option<Vec<i64>>, Option<Vec<&'a Vec<i64>>>) {
            if let Some(bad_word_ids) = bad_word_ids {
                let mut bad_word_ids_length_1 = vec![];
                let mut bad_word_ids_length_greater_than_1 = vec![];
                for bad_word in bad_word_ids {
                    if bad_word.len() == 1 {
                        bad_word_ids_length_1.push(bad_word[0]);
                    } else {
                        bad_word_ids_length_greater_than_1.push(bad_word);
                    }
                }
                let bad_word_ids_length_1 = if !bad_word_ids_length_1.is_empty() {
                    Some(bad_word_ids_length_1)
                } else {
                    None
                };
                let bad_word_ids_length_greater_than_1 =
                    if !bad_word_ids_length_greater_than_1.is_empty() {
                        Some(bad_word_ids_length_greater_than_1)
                    } else {
                        None
                    };
                (bad_word_ids_length_1, bad_word_ids_length_greater_than_1)
            } else {
                (None, None)
            }
        }

        fn tokens_match(&self, prev_tokens: &[i64], tokens: &[i64]) -> bool {
            if tokens.is_empty() {
                true
            } else if tokens.len() > prev_tokens.len() {
                false
            } else {
                &prev_tokens[prev_tokens.len() - tokens.len()..] == tokens
            }
        }

        fn calc_static_bad_word_mask(
            &self,
            scores: &Array2<f32>,
            bad_words_id_length_1: &[i64],
        ) -> Array2<bool> {
            let mut mask = Array2::<bool>::from_elem((1, scores.ncols()), false);
            for &token in bad_words_id_length_1 {
                if (token as usize) < scores.ncols() {
                    mask[[0, token as usize]] = true;
                }
            }
            mask
        }

        fn get_dynamic_bad_word_ids(
            &self,
            prev_tokens: &[Vec<i64>],
            bad_word_ids_length_greater_than_1: &[&Vec<i64>],
        ) -> Vec<Vec<i64>> {
            let mut banned_tokens = Vec::new();
            for prev_token_sequence in prev_tokens {
                let mut sequence_banned_tokens = Vec::new();
                for bad_word_ids in bad_word_ids_length_greater_than_1 {
                    if self
                        .tokens_match(prev_token_sequence, &bad_word_ids[..bad_word_ids.len() - 1])
                    {
                        sequence_banned_tokens.push(*bad_word_ids.last().unwrap());
                    }
                }
                banned_tokens.push(sequence_banned_tokens);
            }

            banned_tokens
        }

        fn ban_bad_words(
            &self,
            dynamic_bad_words: Option<&Vec<&Vec<i64>>>,
            static_bad_words_mask: Option<&Array2<bool>>,
            token_ids: &Array2<i64>,
            scores: &mut Array2<f32>,
        ) {
            let longest_bad_word = dynamic_bad_words
                .iter()
                .flat_map(|bad_words| bad_words.iter())
                .map(|bad_word| bad_word.len())
                .max()
                .unwrap_or(1) as i64;

            let mut prev_tokens = Vec::new();
            for row in token_ids.rows() {
                let start = row.len().saturating_sub(longest_bad_word as usize);
                prev_tokens.push(row.slice(ndarray::s![start..]).to_vec());
            }

            let dynamic_bad_words_mask = if let Some(dynamic_bad_words) = dynamic_bad_words {
                let dynamic_banned_tokens =
                    self.get_dynamic_bad_word_ids(&prev_tokens, dynamic_bad_words);
                let mut mask = Array2::<bool>::from_elem((scores.nrows(), scores.ncols()), false);
                for (sequence_index, sequence_ban_tokens) in
                    dynamic_banned_tokens.iter().enumerate()
                {
                    if sequence_index >= scores.nrows() {
                        break;
                    }
                    for &token in sequence_ban_tokens {
                        if (token as usize) < scores.ncols() {
                            mask[[sequence_index, token as usize]] = true;
                        }
                    }
                }
                Some(mask)
            } else {
                None
            };

            let combined_bad_word_mask = {
                if let (Some(static_mask), Some(dynamic_mask)) =
                    (static_bad_words_mask, &dynamic_bad_words_mask)
                {
                    Some(static_mask | dynamic_mask)
                } else {
                    None
                }
            };

            let bad_word_mask = if combined_bad_word_mask.is_some() {
                combined_bad_word_mask.as_ref()
            } else if static_bad_words_mask.is_some() {
                static_bad_words_mask
            } else if dynamic_bad_words_mask.is_some() {
                dynamic_bad_words_mask.as_ref()
            } else {
                None
            };

            if let Some(bad_word_mask) = bad_word_mask {
                for (logit, &flag) in scores.iter_mut().zip(bad_word_mask.iter()) {
                    if flag {
                        *logit = NEG_INF;
                    }
                }
            }
        }

        fn generate_no_beam_search(
            &self,
            input_ids: Array2<i64>,
            encoder_outputs: Option<ArrayD<f32>>,
            cur_len: i64,
            _batch_size: i64,
            attention_mask: Array2<i64>,
            gen_opt: InternalGenerateOptions,
            prefix_allowed_tokens_fn: Option<PrefixAllowedFunction>,
            output_scores: bool,
        ) -> GeneratedOutputWithScores {
            let mut unfinished_sentences = vec![1i64; input_ids.nrows()];
            let mut sentence_lengths = vec![1i64; input_ids.nrows()];
            let (bad_word_ids_length_1, bad_word_ids_length_greater_than_1) =
                self.split_bad_word_ids(gen_opt.bad_word_ids);
            let mut static_bad_words_mask: Option<Array2<bool>> = None;
            let mut attention_mask = attention_mask;
            let mut input_ids = input_ids;
            let mut past: Cache = Cache::None;
            let mut current_length = cur_len;
            let mut token_scores_output: Option<Vec<Vec<f64>>> =
                if output_scores { Some(vec![]) } else { None };

            loop {
                let prepared_input = self.prepare_inputs_for_generation(
                    input_ids.clone(),
                    encoder_outputs.as_ref(),
                    past,
                    attention_mask.clone(),
                );
                let temp = self
                    .forward_t(
                        prepared_input.prepared_input.as_ref(),
                        prepared_input.prepared_past,
                        prepared_input.prepared_attention_mask.as_ref(),
                        None,
                        prepared_input.prepared_position_ids.as_ref(),
                        None,
                        prepared_input.prepared_encoder_output,
                        prepared_input.prepared_decoder_input.as_ref(),
                        false,
                    )
                    .unwrap();
                past = temp.cache;

                // Only the last position logits are needed
                let mut next_token_logits = last_step_logits(&temp.lm_logits);

                // Reduce probability for repeated inputs
                if gen_opt.repetition_penalty > 1f64 {
                    self.enforce_repetition_penalty(
                        &mut next_token_logits,
                        input_ids.nrows() as i64,
                        1,
                        &input_ids,
                        gen_opt.repetition_penalty,
                    )
                }

                // Get bad word_ids and set their probability to 0
                if gen_opt.bad_word_ids.is_some() {
                    // Calculate static bad words masks if not set yet
                    if let Some(bad_word_ids_length_1) = &bad_word_ids_length_1 {
                        if static_bad_words_mask.is_none() {
                            static_bad_words_mask = Some(self.calc_static_bad_word_mask(
                                &next_token_logits,
                                bad_word_ids_length_1,
                            ));
                        }
                    }
                    self.ban_bad_words(
                        bad_word_ids_length_greater_than_1.as_ref(),
                        static_bad_words_mask.as_ref(),
                        &input_ids,
                        &mut next_token_logits,
                    );
                }

                // Get banned tokens and set their probability to 0
                if gen_opt.no_repeat_ngram_size > 0 {
                    let banned_tokens = self.get_banned_tokens(
                        &input_ids,
                        gen_opt.no_repeat_ngram_size,
                        current_length,
                    );
                    for (batch_index, index_banned_token) in banned_tokens.into_iter().enumerate() {
                        for token in index_banned_token {
                            if (token as usize) < next_token_logits.ncols() {
                                next_token_logits[[batch_index, token as usize]] = NEG_INF;
                            }
                        }
                    }
                }

                // Apply custom prefix constraint function
                if let Some(prefix_allowed_tokens_function) = prefix_allowed_tokens_fn {
                    self.apply_prefix_allowed_tokens_function(
                        prefix_allowed_tokens_function,
                        1,
                        &input_ids,
                        &mut next_token_logits,
                    )
                }

                // Do not allow eos token if min length is not reached
                if (gen_opt.eos_token_ids.is_some()) & (current_length < gen_opt.min_length) {
                    for &eos_token_id in gen_opt.eos_token_ids.as_ref().unwrap() {
                        for mut row in next_token_logits.rows_mut().into_iter() {
                            row[eos_token_id as usize] = NEG_INF;
                        }
                    }
                }

                self.prepare_scores_for_generation(
                    &mut next_token_logits,
                    current_length,
                    gen_opt.max_length,
                    gen_opt.forced_bos_token_id,
                );

                // Top-k and top-p sampling
                let next_tokens: Vec<i64> = if gen_opt.do_sample {
                    if gen_opt.temperature > 1f64 {
                        next_token_logits.mapv_inplace(|value| value / gen_opt.temperature as f32);
                    }
                    self.top_k_top_p_filtering(
                        &mut next_token_logits,
                        gen_opt.top_k,
                        gen_opt.top_p,
                        1,
                    );
                    let probabilities = crate::common::tensor_ops::softmax_last_dim(
                        &next_token_logits.clone().into_dyn(),
                    )
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                    probabilities
                        .rows()
                        .into_iter()
                        .map(|row| {
                            let weights: Vec<f32> = row.iter().copied().collect();
                            weighted_sample_without_replacement(&weights, 1)[0] as i64
                        })
                        .collect()
                } else {
                    next_token_logits
                        .rows()
                        .into_iter()
                        .map(|row| {
                            let mut best = 0usize;
                            let mut best_value = f32::NEG_INFINITY;
                            for (index, &value) in row.iter().enumerate() {
                                if value > best_value {
                                    best_value = value;
                                    best = index;
                                }
                            }
                            best as i64
                        })
                        .collect()
                };

                if let Some(prev_scores) = token_scores_output.as_mut() {
                    let log_probs = crate::common::tensor_ops::log_softmax_last_dim(
                        &next_token_logits.clone().into_dyn(),
                    )
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                    let step_scores: Vec<f64> = next_tokens
                        .iter()
                        .enumerate()
                        .map(|(row, &token)| {
                            if unfinished_sentences[row] == 0 {
                                0f64
                            } else {
                                log_probs[[row, token as usize]] as f64
                            }
                        })
                        .collect();
                    prev_scores.push(step_scores);
                };

                // Add tokens to unfinished sentences
                let tokens_to_add: Vec<i64> = match &gen_opt.eos_token_ids {
                    Some(_) => next_tokens
                        .iter()
                        .zip(unfinished_sentences.iter())
                        .map(|(&token, &unfinished)| {
                            token * unfinished - gen_opt.pad_token_id.unwrap() * (unfinished - 1)
                        })
                        .collect(),
                    None => next_tokens,
                };

                input_ids = append_column(&input_ids, &tokens_to_add);
                if gen_opt.eos_token_ids.is_some() {
                    for eos_token_id in gen_opt.eos_token_ids.as_ref().unwrap() {
                        for row in 0..unfinished_sentences.len() {
                            let sentence_with_eos = (tokens_to_add[row] == *eos_token_id) as i64
                                * unfinished_sentences[row];
                            if sentence_with_eos != 0 {
                                sentence_lengths[row] = current_length + 1;
                            }
                            unfinished_sentences[row] =
                                -unfinished_sentences[row] * (sentence_with_eos - 1);
                        }
                    }
                    if unfinished_sentences.iter().max().copied().unwrap_or(0) == 0 {
                        break;
                    }
                }
                if !self.is_encoder_decoder() {
                    let ones_column = vec![1i64; attention_mask.nrows()];
                    attention_mask = append_column(&attention_mask, &ones_column);
                }
                current_length += 1;
                if let Some(max_length) = gen_opt.max_length {
                    if current_length >= max_length {
                        for row in 0..sentence_lengths.len() {
                            if unfinished_sentences[row] != 0 {
                                sentence_lengths[row] = current_length;
                            }
                        }
                        break;
                    }
                }
            }
            let scores_output = token_scores_output.as_ref().map(|scores_steps| {
                let per_sequence: Vec<f64> = (0..input_ids.nrows())
                    .map(|row| {
                        scores_steps
                            .iter()
                            .map(|step| step.get(row).copied().unwrap_or(0f64))
                            .sum::<f64>()
                    })
                    .collect();
                per_sequence
                    .into_iter()
                    .zip(sentence_lengths.iter())
                    .map(|(score, &length)| score / (length as f64).powf(gen_opt.length_penalty))
                    .collect()
            });
            let token_scores_output = token_scores_output.map(|scores_steps| {
                (0..input_ids.nrows())
                    .map(|row| {
                        scores_steps
                            .iter()
                            .map(|step| step.get(row).copied().unwrap_or(0f64))
                            .collect()
                    })
                    .collect()
            });
            GeneratedOutputWithScores {
                indices: input_ids,
                scores: scores_output,
                token_scores: token_scores_output,
            }
        }

        fn generate_beam_search(
            &self,
            mut input_ids: Array2<i64>,
            mut encoder_outputs: Option<ArrayD<f32>>,
            cur_len: i64,
            batch_size: i64,
            mut attention_mask: Array2<i64>,
            gen_opt: InternalGenerateOptions,
            prefix_allowed_tokens_fn: Option<PrefixAllowedFunction>,
            output_scores: bool,
        ) -> GeneratedOutputWithScores {
            let num_beam_groups = gen_opt.num_beam_groups.unwrap_or(1);
            let num_sub_beams = gen_opt.num_beams / num_beam_groups;
            let diversity_penalty = gen_opt.diversity_penalty.unwrap_or(5.5) as f32;
            let (bad_word_ids_length_1, bad_word_ids_length_greater_than_1) =
                self.split_bad_word_ids(gen_opt.bad_word_ids);
            let mut static_bad_words_mask: Option<Array2<bool>> = None;

            let mut hypotheses = (0..batch_size)
                .map(|_| {
                    BeamHypotheses::new(
                        gen_opt.num_beams,
                        gen_opt.max_length,
                        gen_opt.length_penalty,
                        gen_opt.early_stopping,
                    )
                })
                .collect::<Vec<BeamHypotheses>>();

            let vocab_size = self.get_vocab_size();
            // Beam scores: initialized to -1e9 except for the first beam of each group
            let mut beam_scores: Vec<f32> = vec![-1e9; (batch_size * gen_opt.num_beams) as usize];
            for beam_index in (0..gen_opt.num_beams).step_by(num_sub_beams as usize) {
                for batch_index in 0..batch_size {
                    beam_scores[(batch_index * gen_opt.num_beams + beam_index) as usize] = 0f32;
                }
            }
            let mut beam_tokens: Vec<i64> = vec![0; (batch_size * gen_opt.num_beams) as usize];
            let mut beam_indices: Vec<i64> = vec![0; (batch_size * gen_opt.num_beams) as usize];
            let mut saved_beam_scores: Option<Vec<Vec<f32>>> =
                if output_scores { Some(vec![]) } else { None };
            let mut current_tokens: Vec<i64> = vec![0; (batch_size * gen_opt.num_beams) as usize];

            let mut past: Cache = Cache::None;
            let mut done = vec![false; batch_size as usize];

            let mut current_length = cur_len;

            loop {
                if num_beam_groups > 1 {
                    current_tokens = vec![0; (batch_size * gen_opt.num_beams) as usize];
                }
                let prepared_input = self.prepare_inputs_for_generation(
                    input_ids.clone(),
                    encoder_outputs.as_ref(),
                    past,
                    attention_mask.clone(),
                );
                let temp = self
                    .forward_t(
                        prepared_input.prepared_input.as_ref(),
                        prepared_input.prepared_past,
                        prepared_input.prepared_attention_mask.as_ref(),
                        None,
                        prepared_input.prepared_position_ids.as_ref(),
                        None,
                        prepared_input.prepared_encoder_output,
                        prepared_input.prepared_decoder_input.as_ref(),
                        false,
                    )
                    .unwrap();
                let outputs = last_step_logits(&temp.lm_logits);
                past = temp.cache;

                for beam_group_index in 0..num_beam_groups {
                    let group_start_index = beam_group_index * num_sub_beams;
                    let group_end_index = min(group_start_index + num_sub_beams, gen_opt.num_beams);
                    let group_size = group_end_index - group_start_index;

                    let batch_group_indices: Vec<i64> = if num_beam_groups > 1 {
                        let mut indices: Vec<i64> =
                            Vec::with_capacity((batch_size * group_size) as usize);
                        for batch_index in 0..batch_size {
                            indices.extend(
                                (group_start_index..group_end_index)
                                    .map(|value| value + batch_index * gen_opt.num_beams),
                            )
                        }
                        indices
                    } else {
                        Vec::new()
                    };

                    let group_input_ids: Array2<i64> = if num_beam_groups > 1 {
                        index_select_rows_i64(&input_ids, &batch_group_indices)
                    } else {
                        input_ids.clone()
                    };

                    let mut next_token_logits = if num_beam_groups <= 1 {
                        outputs.clone()
                    } else {
                        crate::common::tensor_ops::gather_rows(
                            &outputs.clone().into_dyn(),
                            &batch_group_indices,
                        )
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                    };
                    // Reduce probability for repeated inputs
                    if gen_opt.repetition_penalty > 1f64 {
                        self.enforce_repetition_penalty(
                            &mut next_token_logits,
                            batch_size,
                            1,
                            &group_input_ids,
                            gen_opt.repetition_penalty,
                        )
                    }

                    if gen_opt.temperature > 1f64 {
                        next_token_logits.mapv_inplace(|value| value / gen_opt.temperature as f32);
                    }
                    self.prepare_scores_for_generation(
                        &mut next_token_logits,
                        current_length,
                        gen_opt.max_length,
                        gen_opt.forced_bos_token_id,
                    );

                    let mut scores = crate::common::tensor_ops::log_softmax_last_dim(
                        &next_token_logits.clone().into_dyn(),
                    )
                    .into_dimensionality::<Ix2>()
                    .unwrap();

                    // Do not allow eos token if min length is not reached
                    if (gen_opt.eos_token_ids.is_some()) & (current_length < gen_opt.min_length) {
                        for &eos_token_id in gen_opt.eos_token_ids.as_ref().unwrap() {
                            for mut row in scores.rows_mut().into_iter() {
                                row[eos_token_id as usize] = NEG_INF;
                            }
                        }
                    }

                    // Get bad word_ids and set their probability to 0
                    if gen_opt.bad_word_ids.is_some() {
                        // Calculate static bad words masks if not set yet
                        if let Some(bad_word_ids_length_1) = &bad_word_ids_length_1 {
                            if static_bad_words_mask.is_none() {
                                static_bad_words_mask = Some(
                                    self.calc_static_bad_word_mask(&scores, bad_word_ids_length_1),
                                );
                            }
                        }
                        self.ban_bad_words(
                            bad_word_ids_length_greater_than_1.as_ref(),
                            static_bad_words_mask.as_ref(),
                            &group_input_ids,
                            &mut scores,
                        );
                    }

                    // Get repeated tokens and set their probability to 0
                    if gen_opt.no_repeat_ngram_size > 0 {
                        let banned_tokens = self.get_banned_tokens(
                            &group_input_ids,
                            gen_opt.no_repeat_ngram_size,
                            current_length,
                        );
                        for (batch_index, index_banned_token) in
                            banned_tokens.into_iter().enumerate()
                        {
                            for token in index_banned_token {
                                if (token as usize) < scores.ncols() {
                                    scores[[batch_index, token as usize]] = NEG_INF;
                                }
                            }
                        }
                    }

                    // Update scores with diversity penalty
                    if num_beam_groups > 1 {
                        self.run_hamming_diversity_penalty(
                            &mut scores,
                            &Array1::from(current_tokens.clone()),
                            diversity_penalty,
                            gen_opt.num_beams,
                            batch_size,
                            group_size,
                            group_start_index,
                        );
                    }

                    // Apply custom prefix constraint function
                    if let Some(prefix_allowed_tokens_function) = prefix_allowed_tokens_fn {
                        self.apply_prefix_allowed_tokens_function(
                            prefix_allowed_tokens_function,
                            num_sub_beams,
                            &input_ids,
                            &mut scores,
                        )
                    }

                    // Accumulate the running beam score on top of the token scores
                    let mut next_scores = scores.clone();
                    for row in 0..next_scores.nrows() {
                        let beam_score = if num_beam_groups > 1 {
                            beam_scores[batch_group_indices[row] as usize]
                        } else {
                            beam_scores[row]
                        };
                        for value in next_scores.row_mut(row) {
                            *value += beam_score;
                        }
                    }

                    // Select the top (2 * group_size) candidates per batch row
                    let (next_scores_flat, next_tokens_flat): (Vec<Vec<f32>>, Vec<Vec<i64>>) =
                        if gen_opt.do_sample {
                            self.top_k_top_p_filtering(
                                &mut next_scores,
                                gen_opt.top_k,
                                gen_opt.top_p,
                                2,
                            );
                            let flattened: Vec<f32> = next_scores.iter().copied().collect();
                            let mut sampled_tokens_rows = Vec::with_capacity(next_scores.nrows());
                            let mut sampled_scores_rows = Vec::with_capacity(next_scores.nrows());
                            for row in 0..next_scores.nrows() {
                                let row_probs = {
                                    let row_values = &flattened[row * next_scores.ncols()
                                        ..(row + 1) * next_scores.ncols()];
                                    crate::common::tensor_ops::softmax_last_dim(
                                        &ndarray::Array1::from(row_values.to_vec()).into_dyn(),
                                    )
                                    .into_dimensionality::<Ix1>()
                                    .unwrap()
                                };
                                let probs: Vec<f32> = row_probs.iter().copied().collect();
                                let mut sampled = weighted_sample_without_replacement(
                                    &probs,
                                    2 * group_size as usize,
                                );
                                sampled.sort_unstable();
                                let tokens: Vec<i64> =
                                    sampled.iter().map(|&index| index as i64).collect();
                                let values: Vec<f32> = sampled
                                    .iter()
                                    .map(|&index| flattened[row * next_scores.ncols() + index])
                                    .collect();
                                sampled_tokens_rows.push(tokens);
                                sampled_scores_rows.push(values);
                            }
                            (sampled_scores_rows, sampled_tokens_rows)
                        } else {
                            let flattened: Vec<f32> = next_scores.iter().copied().collect();
                            let mut top_tokens_rows = Vec::with_capacity(next_scores.nrows());
                            let mut top_scores_rows = Vec::with_capacity(next_scores.nrows());
                            for row in 0..next_scores.nrows() {
                                let row_values = &flattened
                                    [row * next_scores.ncols()..(row + 1) * next_scores.ncols()];
                                let (values, indices) = crate::common::tensor_ops::topk_last_dim(
                                    &ndarray::Array1::from(row_values.to_vec()).into_dyn(),
                                    2 * group_size as usize,
                                );
                                top_scores_rows.push(
                                    values
                                        .into_dimensionality::<Ix1>()
                                        .unwrap()
                                        .iter()
                                        .copied()
                                        .collect(),
                                );
                                top_tokens_rows.push(
                                    indices
                                        .into_dimensionality::<Ix1>()
                                        .unwrap()
                                        .iter()
                                        .copied()
                                        .collect(),
                                );
                            }
                            (top_scores_rows, top_tokens_rows)
                        };

                    // Beam bookkeeping (eos handling follows the reference implementation)
                    let eos_token_ids = gen_opt.eos_token_ids.as_ref();
                    let mut group_beam_scores: Vec<f32> = Vec::new();
                    let mut group_beam_tokens: Vec<i64> = Vec::new();
                    let mut group_beam_indices: Vec<i64> = Vec::new();
                    for batch_index in 0..batch_size as usize {
                        let row_tokens = &next_tokens_flat[batch_index];
                        let row_scores = &next_scores_flat[batch_index];
                        // effective beam id of each candidate
                        let effective_beam_ids: Vec<i64> = row_tokens
                            .iter()
                            .map(|&token| {
                                let beam_id = token / vocab_size;
                                batch_index as i64 * group_size + beam_id
                            })
                            .collect();
                        let token_ids: Vec<i64> = row_tokens
                            .iter()
                            .map(|&token| token - (token / vocab_size) * vocab_size)
                            .collect();
                        let max_score =
                            row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                        // eos mask: 1 unless the candidate is the (first) eos token
                        let eos_mask: Vec<i64> = match eos_token_ids {
                            Some(ids) => token_ids
                                .iter()
                                .map(|&token| 1 - (token == ids[0]) as i64)
                                .collect(),
                            None => vec![1; row_tokens.len()],
                        };
                        // active candidates: the first group_size candidates that are not eos
                        let mut cumulative = 0i64;
                        let is_active: Vec<bool> = eos_mask
                            .iter()
                            .map(|&flag| {
                                cumulative += flag;
                                (cumulative <= group_size) && flag != 0
                            })
                            .collect();

                        for (candidate_index, &active) in is_active.iter().enumerate() {
                            if active {
                                group_beam_scores.push(row_scores[candidate_index]);
                                group_beam_tokens.push(token_ids[candidate_index]);
                                group_beam_indices.push(effective_beam_ids[candidate_index]);
                            }
                        }

                        // completed hypotheses: candidates where eos_mask == 0
                        for (candidate_index, &flag) in eos_mask.iter().enumerate() {
                            if flag != 0 {
                                continue;
                            }
                            if !done[batch_index] {
                                let beam_index_pos = candidate_index as i64;
                                let is_beam_token_worse_than_top_num_beams =
                                    beam_index_pos >= gen_opt.num_beams;
                                if is_beam_token_worse_than_top_num_beams {
                                    continue;
                                }
                                let effective_beam_id =
                                    effective_beam_ids[candidate_index] as usize;
                                let beam_token_score = row_scores[candidate_index] as f64;
                                let saved_step_scores =
                                    saved_beam_scores.as_ref().map(|step_wise_scores| {
                                        step_wise_scores
                                            .iter()
                                            .map(|step| {
                                                step.get(effective_beam_id).copied().unwrap_or(0f32)
                                                    as f64
                                            })
                                            .collect::<Vec<f64>>()
                                    });
                                hypotheses[batch_index].add(
                                    input_ids
                                        .row(min(effective_beam_id, input_ids.nrows() - 1))
                                        .to_vec(),
                                    beam_token_score,
                                    saved_step_scores,
                                );
                            }
                        }

                        if done[batch_index] {
                            for _ in 0..gen_opt.num_beams {
                                group_beam_scores.push(0f32);
                                group_beam_tokens.push(gen_opt.pad_token_id.unwrap());
                                group_beam_indices.push(0);
                            }
                        } else {
                            done[batch_index] |=
                                hypotheses[batch_index].is_done(max_score as f64, current_length);
                        }
                    }

                    if num_beam_groups <= 1 {
                        beam_scores = group_beam_scores;
                        beam_tokens = group_beam_tokens;
                        beam_indices = group_beam_indices;
                    } else {
                        for (position, &batch_beam_index) in batch_group_indices.iter().enumerate()
                        {
                            let index = batch_beam_index as usize;
                            beam_scores[index] = group_beam_scores[position];
                            beam_tokens[index] = group_beam_tokens[position];
                            let new_index = gen_opt.num_beams
                                * (group_beam_indices[position] / group_size)
                                + group_start_index
                                + (group_beam_indices[position] % group_size);
                            beam_indices[index] = new_index;
                            current_tokens[index] = group_beam_tokens[position];
                        }
                    }
                }

                if let Some(scores_output) = saved_beam_scores.as_mut() {
                    scores_output.push(beam_scores.clone());
                }
                if done.iter().all(|&x| x) {
                    break;
                }

                input_ids = index_select_rows_i64(&input_ids, &beam_indices);
                input_ids = append_column(&input_ids, &beam_tokens);

                current_length += 1;
                if let Some(max_length) = gen_opt.max_length {
                    if current_length >= max_length {
                        break;
                    }
                }
                encoder_outputs = self.reorder_cache(&mut past, encoder_outputs, &beam_indices);

                if !self.is_encoder_decoder() {
                    let ones_column = vec![1i64; attention_mask.nrows()];
                    attention_mask = append_column(&attention_mask, &ones_column);
                }
            }

            let mut batch_index = 0i64;

            let mut saved_beam_scores = saved_beam_scores.map(|step_wise_scores| {
                // transpose to per-beam token score sequences
                (0..step_wise_scores.first().map(|step| step.len()).unwrap_or(0))
                    .map(|beam| {
                        step_wise_scores
                            .iter()
                            .map(|step| step.get(beam).copied().unwrap_or(0f32))
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<Vec<f32>>>()
            });
            loop {
                if batch_index == batch_size {
                    break;
                }
                if done[batch_index as usize] {
                    batch_index += 1;
                    continue;
                }
                for beam_index in 0..gen_opt.num_beams {
                    let effective_beam_id = batch_index * gen_opt.num_beams + beam_index;
                    let beam_saved_token_scores = saved_beam_scores.as_mut().map(|saved_tokens| {
                        saved_tokens[effective_beam_id as usize]
                            .iter()
                            .map(|&value| value as f64)
                            .collect::<Vec<f64>>()
                    });
                    let final_score = beam_scores[effective_beam_id as usize] as f64;
                    let final_tokens = input_ids.row(effective_beam_id as usize).to_vec();
                    hypotheses[batch_index as usize].add(
                        final_tokens,
                        final_score,
                        beam_saved_token_scores,
                    );
                }
                batch_index += 1;
            }
            let (output_batch_size, output_num_return_sequences_per_batch) = if gen_opt.do_sample {
                (batch_size, 1)
            } else {
                (
                    batch_size * gen_opt.num_return_sequences,
                    gen_opt.num_return_sequences,
                )
            };

            let mut sentence_lengths: Vec<i64> = vec![0; output_batch_size as usize];
            let mut best_ids: Vec<Vec<i64>> = Vec::new();
            let mut scores_output = if output_scores {
                Some(Vec::with_capacity(best_ids.len()))
            } else {
                None
            };
            let mut token_scores_output = if output_scores {
                Some(Vec::with_capacity(best_ids.len()))
            } else {
                None
            };
            for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
                let mut sorted_hypotheses = hypothesis.clone();
                sorted_hypotheses
                    .beams
                    .sort_by_key(|(score, _, _)| OrderedFloat(*score));
                for j in 0..output_num_return_sequences_per_batch {
                    let effective_batch_index =
                        output_num_return_sequences_per_batch * hypothesis_index as i64 + j;

                    let (best_score, best_hyp, best_token_scores) =
                        sorted_hypotheses.beams.pop().unwrap();
                    sentence_lengths[effective_batch_index as usize] = best_hyp.len() as i64;
                    best_ids.push(best_hyp);
                    if let Some(current_best_scores) = &mut scores_output {
                        current_best_scores.push(best_score);
                    }
                    if let Some(current_best_token_scores) = &mut token_scores_output {
                        current_best_token_scores.push(best_token_scores.unwrap_or_default());
                    }
                }
            }
            let max_sentence_length = sentence_lengths.iter().max().copied().unwrap_or(0);
            let sentence_max_length = gen_opt
                .max_length
                .map(|max_length| min(max_sentence_length + 1, max_length))
                .unwrap_or(max_sentence_length + 1);

            let mut decoded = ndarray::Array2::<i64>::from_elem(
                (output_batch_size as usize, sentence_max_length as usize),
                gen_opt
                    .pad_token_id
                    .unwrap_or_else(|| gen_opt.eos_token_ids.as_ref().unwrap()[0]),
            );
            for (hypothesis_index, best_id) in best_ids.iter().enumerate() {
                let sentence_length = sentence_lengths[hypothesis_index] as usize;
                for (position, &token) in best_id.iter().take(sentence_length).enumerate() {
                    decoded[[hypothesis_index, position]] = token;
                }
                let sentence_length_max = gen_opt.max_length.unwrap_or(max_sentence_length);
                if (sentence_length as i64) < sentence_length_max
                    && sentence_length < decoded.ncols()
                {
                    decoded[[hypothesis_index, sentence_length]] =
                        gen_opt.eos_token_ids.as_ref().unwrap()[0];
                }
            }
            GeneratedOutputWithScores {
                indices: decoded,
                scores: scores_output,
                token_scores: token_scores_output,
            }
        }

        fn reorder_cache(
            &self,
            past: &mut Cache,
            _encoder_outputs: Option<ArrayD<f32>>,
            _beam_indices: &[i64],
        ) -> Option<ArrayD<f32>> {
            match past {
                Cache::None => None,
                _ => {
                    panic!("Not implemented");
                }
            }
        }
    }

    fn get_positive_infinity_f32() -> f32 {
        f32::INFINITY
    }

    /// Extracts the logits of the last sequence position, whatever the rank of the model output.
    fn last_step_logits(logits: &ArrayD<f32>) -> Array2<f32> {
        match logits.ndim() {
            2 => logits.clone().into_dimensionality::<Ix2>().unwrap(),
            3 => {
                let batch = logits.shape()[0];
                let seq = logits.shape()[1];
                let vocab = logits.shape()[2];
                let view = logits.view();
                let mut selected = ndarray::Array2::<f32>::zeros((batch, vocab));
                for row in 0..batch {
                    for col in 0..vocab {
                        selected[[row, col]] = view[[row, seq - 1, col]];
                    }
                }
                selected
            }
            _rank => panic!("Unexpected logits rank"),
        }
    }

    pub fn force_token_id_generation(scores: &mut Array2<f32>, token_ids: &[i64], vocab_size: i64) {
        let impossible_tokens: Vec<i64> = (0..vocab_size)
            .filter(|pos| !token_ids.contains(pos))
            .collect();
        for mut row in scores.rows_mut().into_iter() {
            for token in &impossible_tokens {
                row[*token as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

#[derive(Debug, Clone)]
/// # Generated text output
/// Contains generated text and an optional log-likelihood score for the generated sequence
pub struct GeneratedTextOutput {
    pub text: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone)]
/// # Generated indices output
/// Contains generated indices and an optional log-likelihood score for the generated sequence and individual tokens
pub struct GeneratedIndicesOutput {
    pub indices: Vec<i64>,
    pub score: Option<f64>,
    pub token_scores: Option<Vec<f64>>,
}

pub type PrefixAllowedFunction<'a> = &'a dyn Fn(i64, &[i64]) -> Vec<i64>;
/// Type alias for a function defining allowed tokens based on current tokens generated.
/// This function should take a `batch_id` and associated slice of already generated tokens and
/// should return a vector of allowed tokens. This is useful for controlled generation, i.e.
/// deterministic generation of a token continuation if a sequence of token occurs.

#[derive(Clone, Copy, Default)]
/// # Generation options for text generation.
/// When provided to a `generate` method, these options will take priority over the `GenerateConfig` used to create the
/// `LanguageGenerator`. Some of these options may be left as `None`, options without a value will individually default
/// to the `GenerateConfig`.
pub struct GenerateOptions<'a> {
    /// Minimum sequence length
    pub min_length: Option<i64>,
    /// Maximum sequence length
    pub max_length: Option<i64>,
    /// Maximum number of new tokens to generate (useful for causal generation models).
    /// Only one of `max_length` and `max_new_tokens` should be provided.
    /// When both are given, `max_new_tokens` is ignored and the `max_length` setting is used.
    pub max_new_tokens: Option<i64>,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated
    pub early_stopping: Option<bool>,
    /// Number of sequences to return for each prompt text
    pub num_return_sequences: Option<i64>,
    /// Number of beams for beam search
    pub num_beams: Option<i64>,
    pub num_beam_groups: Option<i64>,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding
    pub do_sample: Option<bool>,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance
    pub temperature: Option<f64>,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature
    pub top_k: Option<i64>,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p
    pub top_p: Option<f64>,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated.
    pub repetition_penalty: Option<f64>,
    /// Exponential penalty based on the length of the hypotheses generated
    pub length_penalty: Option<f64>,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature
    pub no_repeat_ngram_size: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups
    pub diversity_penalty: Option<f64>,
    /// Decoder start token id
    pub decoder_start_token_id: Option<i64>,
    /// Forced first token generated
    pub forced_bos_token_id: Option<i64>,
    /// Function to control the generation process. The function should take a `batch_id` (i64) and a slice of token_ids already generated and returns a `Vec<i64>` of allowed tokens.
    pub prefix_allowed_tokens_fn: Option<PrefixAllowedFunction<'a>>,
    /// List of bad word ids (may be a sequence of word ids) that will be banned during the generation
    pub bad_word_ids: Option<&'a Vec<Vec<i64>>>,
    /// Flag indicating if text generation scores should be returned
    pub output_scores: bool,
}

macro_rules! unpack_config {
    ($field_name:ident, $generate_options: ident, $generate_config: ident) => {
        $generate_options.map_or($generate_config.$field_name, |opts| {
            opts.$field_name.unwrap_or($generate_config.$field_name)
        })
    };
}

/// # Common trait for text generation models.
/// Main API for text generation
pub trait LanguageGenerator: PrivateLanguageGenerator {
    /// Generate text based on a vector of promp texts.
    ///
    /// # Arguments
    ///
    /// * `prompt_texts` - `Option<Vec<&str>>` Optional vector of text prompts. An empty prompt to the model may be passed if the model implement a `bos_id`.
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<TextOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing TextOutput with the generated texts and the generation score if `output_scores` is true.
    fn generate<S>(
        &self,
        prompt_texts: Option<&[S]>,
        generate_options: Option<GenerateOptions>,
    ) -> Result<Vec<GeneratedTextOutput>, RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        let indices_outputs = self.generate_indices(prompt_texts, generate_options)?;
        let mut output = Vec::with_capacity(indices_outputs.len());
        for generated_sequence in indices_outputs {
            output.push(GeneratedTextOutput {
                text: self
                    ._get_tokenizer()
                    .decode(&generated_sequence.indices, true, true),
                score: generated_sequence.score,
            });
        }
        Ok(output)
    }

    /// Generate token indices without decoding (useful for token-level operations before returning final text or as validation step during training).
    ///
    /// # Arguments
    ///
    /// * `prompt_texts` - `Option<Vec<&str>>` Optional vector of text prompts. An empty prompt to the model may be passed if the model implement a `bos_id`.
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<IndicesOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing IndicesOutput with the generated indices and the generation score if `output_scores` is true.
    fn generate_indices<S>(
        &self,
        prompt_texts: Option<&[S]>,
        generate_options: Option<GenerateOptions>,
    ) -> Result<Vec<GeneratedIndicesOutput>, RustBertError>
    where
        S: AsRef<str> + Send + Sync,
    {
        let eos_token_ids = self.get_eos_ids();

        let config = self.get_config();

        let max_length = generate_options.map_or(config.max_length, |generate_options| {
            generate_options.max_length
        });
        let encoding_max_len = if self.is_encoder_decoder() {
            self.get_max_positions_embeddings()
        } else {
            max_length
        };
        let pad_token_id = match self.get_pad_id() {
            Some(value) => Some(value),
            None => eos_token_ids.as_ref().map(|eos_ids| eos_ids[0]),
        };

        let input_ids = match prompt_texts {
            Some(prompts) if !prompts.is_empty() => {
                self.encode_prompt_text(prompts, encoding_max_len, pad_token_id)
            }
            None => match self.get_bos_id() {
                Some(bos_id) => ndarray::Array2::from_elem((1, 1), bos_id),
                None => return Err(RustBertError::ValueError(
                    "A model with a BOS token must be used to start generation with an empty input"
                        .to_string(),
                )),
            },
            _ => return Ok(Vec::new()),
        };
        self.generate_from_ids_and_past(input_ids, None, generate_options)
    }

    /// Generate token indices given a list of indices (useful when the input has been pre-tokenized).
    /// Returns a list of output tokens that need to be decoded using a tokenizer.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `Array2<i64>` pre-tokenized and encoded input for generation (shape *batch size* x *sequence length*).
    /// * `generate_options` - `Option<GenerateOptions>` Optional set of generate options. If not (or partially) provided, will use the settings provided when creating the generator
    ///
    /// # Returns
    /// * `Vec<IndicesOutput>` Vector of length *number_of_prompts* x *num_return_sequences* containing IndicesOutput with the generated indices and the generation score if `output_scores` is true.
    fn generate_from_ids_and_past(
        &self,
        mut input_ids: ndarray::Array2<i64>,
        mut attention_mask: Option<ndarray::Array2<i64>>,
        generate_options: Option<GenerateOptions>,
    ) -> Result<Vec<GeneratedIndicesOutput>, RustBertError> {
        let eos_token_ids = PrivateLanguageGenerator::get_eos_ids(self).cloned();

        let config = PrivateLanguageGenerator::get_config(self);

        // Set generation options. Priority goes to options provided to the `generate` method, then
        // model configuration, then default values.
        let do_sample = unpack_config!(do_sample, generate_options, config);
        let num_return_sequences = unpack_config!(num_return_sequences, generate_options, config);
        let num_beams = unpack_config!(num_beams, generate_options, config);
        let min_length = unpack_config!(min_length, generate_options, config);
        let early_stopping = unpack_config!(early_stopping, generate_options, config);
        let temperature = unpack_config!(temperature, generate_options, config);
        let top_k = unpack_config!(top_k, generate_options, config);
        let top_p = unpack_config!(top_p, generate_options, config);
        let repetition_penalty = unpack_config!(repetition_penalty, generate_options, config);
        let length_penalty = unpack_config!(length_penalty, generate_options, config);
        let no_repeat_ngram_size = unpack_config!(no_repeat_ngram_size, generate_options, config);
        let num_beam_groups = generate_options.map_or(config.num_beam_groups, |opts| {
            opts.num_beam_groups.or(config.num_beam_groups)
        });
        let diversity_penalty = generate_options.map_or(config.diversity_penalty, |opts| {
            opts.diversity_penalty.or(config.diversity_penalty)
        });
        let decoder_start_token_id = generate_options.and_then(|opts| opts.decoder_start_token_id);
        let forced_bos_token_id = generate_options.and_then(|opts| opts.forced_bos_token_id);
        let bad_word_ids = generate_options.and_then(|opts| opts.bad_word_ids);
        let prefix_allowed_tokens_fn =
            generate_options.and_then(|opts| opts.prefix_allowed_tokens_fn);
        let output_scores = generate_options.is_some_and(|opts| opts.output_scores);

        let pad_token_id = match self.get_pad_id() {
            Some(value) => Some(value),
            None => eos_token_ids.as_ref().map(|eos_ids| eos_ids[0]),
        };

        let mut input_ids_len = input_ids.ncols() as i64;
        if input_ids_len == 0 {
            input_ids = ndarray::Array2::from_elem(
                (input_ids.nrows(), 1),
                self.get_bos_id()
                    .expect("`bos_token_id` has to be defined when no `input_ids` are provided."),
            );
            attention_mask = Some(ndarray::Array2::ones((input_ids.nrows(), 1)));
            input_ids_len += 1;
        }

        let cur_len = if !self.is_encoder_decoder() {
            input_ids.ncols() as i64
        } else {
            1
        };
        let batch_size = input_ids.nrows() as i64;

        let (effective_batch_size, effective_batch_mult) = match do_sample {
            true => (batch_size * num_return_sequences, num_return_sequences),
            false => (batch_size, 1),
        };

        let attention_mask = match attention_mask {
            Some(value) => value,
            None => match pad_token_id {
                Some(pad_id) => input_ids.mapv(|value| (value != pad_id) as i64),
                None => input_ids.mapv(|_| 1),
            },
        };

        let encoder_outputs = if self.is_encoder_decoder() {
            let encoder_outputs = self
                .encode(&input_ids, Some(&attention_mask))
                .ok_or(RustBertError::UnsupportedError)?;
            let mut expanded_batch_indices: Vec<i64> = Vec::new();
            for batch_index in 0..batch_size {
                for _ in 0..(num_beams * effective_batch_mult) {
                    expanded_batch_indices.push(batch_index);
                }
            }
            Some(crate::common::tensor_ops::gather_rows(
                &encoder_outputs,
                &expanded_batch_indices,
            ))
        } else {
            None
        };

        let (input_ids, attention_mask) = if !self.is_encoder_decoder() {
            if (num_return_sequences > 1) | (num_beams > 1) {
                let total_rows = (effective_batch_size * num_beams) as usize;
                let mut expanded_ids =
                    ndarray::Array2::<i64>::zeros((total_rows, cur_len as usize));
                let mut expanded_mask =
                    ndarray::Array2::<i64>::zeros((total_rows, cur_len as usize));
                for row in 0..total_rows {
                    expanded_ids
                        .row_mut(row)
                        .assign(&input_ids.row(row % batch_size as usize));
                    expanded_mask
                        .row_mut(row)
                        .assign(&attention_mask.row(row % batch_size as usize));
                }
                (expanded_ids, expanded_mask)
            } else {
                (input_ids, attention_mask)
            }
        } else {
            let decoder_start_token_id = decoder_start_token_id
                .or(self.get_decoder_start_id())
                .ok_or(RustBertError::ValueError(
                    "decoder start id must be specified for encoder decoders".to_string(),
                ))?;
            let input_ids = ndarray::Array2::from_elem(
                ((effective_batch_size * num_beams) as usize, 1),
                decoder_start_token_id,
            );
            let attention_mask = if (num_return_sequences > 1) | (num_beams > 1) {
                let total_rows = (effective_batch_size * num_beams) as usize;
                let mut expanded_mask =
                    ndarray::Array2::<i64>::zeros((total_rows, input_ids_len as usize));
                for row in 0..total_rows {
                    expanded_mask
                        .row_mut(row)
                        .assign(&attention_mask.row(row % batch_size as usize));
                }
                expanded_mask
            } else {
                attention_mask
            };
            (input_ids, attention_mask)
        };

        let max_length = if let Some(generate_options) = generate_options {
            match (generate_options.max_length, generate_options.max_new_tokens) {
                (Some(max_length), _) => Some(max_length),
                (None, Some(max_new_tokens)) => Some(max_new_tokens + input_ids.ncols() as i64),
                (None, None) => config.max_length,
            }
        } else {
            config.max_length
        };

        if let Some(max_length) = max_length {
            if input_ids.ncols() as i64 > max_length {
                return Err(RustBertError::ValueError("The input ids exceeds the maximum length for generation.\
                 Reduce the size of the provided input ids or increase the allowable maximum generation length.".to_string()));
            }
        }

        if max_length.is_none() & eos_token_ids.is_none() {
            return Err(RustBertError::InvalidConfigurationError("No maximum length given for a model without an EOS token. \
            This would lead to an infinite generation loop. Please provide a `max_length` or `max_new_tokens`".to_string()));
        }

        let gen_opt = InternalGenerateOptions {
            min_length,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
            pad_token_id,
            eos_token_ids,
            num_return_sequences,
            early_stopping,
            num_beams,
            length_penalty,
            num_beam_groups,
            diversity_penalty,
            forced_bos_token_id,
            bad_word_ids,
        };

        let generated_output_with_scores = private_generation_utils::no_grad(|| {
            if num_beams > 1 {
                self.generate_beam_search(
                    input_ids,
                    encoder_outputs,
                    cur_len,
                    effective_batch_size,
                    attention_mask,
                    gen_opt,
                    prefix_allowed_tokens_fn,
                    output_scores,
                )
            } else {
                self.generate_no_beam_search(
                    input_ids,
                    encoder_outputs,
                    cur_len,
                    effective_batch_size,
                    attention_mask,
                    gen_opt,
                    prefix_allowed_tokens_fn,
                    output_scores,
                )
            }
        });
        let (decoded, scores, mut token_scores) = (
            generated_output_with_scores.indices,
            generated_output_with_scores.scores,
            generated_output_with_scores.token_scores,
        );
        let num_sequences = decoded.nrows();
        let mut output = Vec::with_capacity(num_sequences);
        for sequence_index in 0..num_sequences {
            let indices = decoded.row(sequence_index).to_vec();
            let score = scores
                .as_ref()
                .map(|scores_value| scores_value[sequence_index]);

            let token_scores = token_scores
                .as_mut()
                .map(|token_scores| std::mem::take(&mut token_scores[sequence_index]));

            output.push(GeneratedIndicesOutput {
                indices,
                score,
                token_scores,
            });
        }
        Ok(output)
    }

    /// Returns a reference to the text generator's tokenizer
    ///
    /// # Returns
    /// * `&TokenizerOption` Reference to the generator's tokenizer.
    fn get_tokenizer(&self) -> &TokenizerOption {
        self._get_tokenizer()
    }

    fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        self._get_tokenizer_mut()
    }

    #[cfg(feature = "libtorch")]
    fn half(&mut self) -> Result<(), RustBertError> {
        self.get_var_store_mut()?.half();
        Ok(())
    }

    #[cfg(feature = "libtorch")]
    fn float(&mut self) -> Result<(), RustBertError> {
        self.get_var_store_mut()?.float();
        Ok(())
    }

    #[cfg(feature = "libtorch")]
    fn set_device(&mut self, device: tch::Device) -> Result<(), RustBertError> {
        self.get_var_store_mut()?.set_device(device);
        Ok(())
    }
}

#[derive(Debug)]
struct BeamHypotheses {
    max_length: Option<i64>,
    length_penalty: f64,
    early_stopping: bool,
    num_beams: i64,
    beams: Vec<(f64, Vec<i64>, Option<Vec<f64>>)>,
    worst_score: f64,
}

impl Clone for BeamHypotheses {
    fn clone(&self) -> Self {
        BeamHypotheses {
            max_length: self.max_length,
            length_penalty: self.length_penalty,
            early_stopping: self.early_stopping,
            num_beams: self.num_beams,
            beams: self.beams.clone(),
            worst_score: self.worst_score,
        }
    }
}

impl BeamHypotheses {
    fn new(
        num_beams: i64,
        max_length: Option<i64>,
        length_penalty: f64,
        early_stopping: bool,
    ) -> BeamHypotheses {
        BeamHypotheses {
            max_length: max_length.map(|max_length| max_length - 1),
            length_penalty,
            early_stopping,
            num_beams,
            beams: Vec::with_capacity(num_beams as usize + 1),
            worst_score: 1e9f64,
        }
    }

    fn len(&self) -> i64 {
        self.beams.len() as i64
    }

    fn add(
        &mut self,
        hypothesis: Vec<i64>,
        sum_log_probabilities: f64,
        token_scores: Option<Vec<f64>>,
    ) {
        let score = sum_log_probabilities / ((hypothesis.len() as f64).powf(self.length_penalty));
        if (self.len() < self.num_beams) | (score > self.worst_score) {
            let token_scores = token_scores.map(|scores| {
                // first-order difference with a 0 prepended (matching torch `diff` with prepend)
                let mut diff = Vec::with_capacity(scores.len());
                let mut previous = 0f64;
                for value in scores {
                    diff.push(value - previous);
                    previous = value;
                }
                diff
            });
            self.beams.push((score, hypothesis, token_scores));
            if self.len() > self.num_beams {
                let (worst_score_position, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (score, _, _))| OrderedFloat(*score))
                    .unwrap();
                let _ = self.beams.remove(worst_score_position);
            }
            self.worst_score = self
                .beams
                .iter()
                .min_by_key(|(score, _, _)| OrderedFloat(*score))
                .unwrap()
                .0;
        }
    }

    fn is_done(&self, best_sum_log_probabilities: f64, current_length: i64) -> bool {
        if self.len() < self.num_beams {
            false
        } else if self.early_stopping {
            true
        } else {
            self.worst_score
                >= best_sum_log_probabilities / (current_length as f64).powf(self.length_penalty)
        }
    }
}

/// Container holding a language model output for generation tasks (LibTorch backend).
#[cfg(feature = "libtorch")]
pub struct LMModelOutput {
    /// Logits for each vocab item and position
    pub lm_logits: tch::Tensor,
    /// cached state for improved efficiency during decoding
    pub cache: Cache,
}

/// Container holding a language model output for generation tasks (backend-neutral).
pub struct GeneratedLogits {
    /// Logits for each vocab item and position
    pub lm_logits: ArrayD<f32>,
    /// cached state for improved efficiency during decoding
    pub cache: Cache,
}
