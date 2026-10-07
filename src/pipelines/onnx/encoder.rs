use crate::pipelines::onnx::common::{get_input_output_mapping, InputOutputNameMapping};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, END_LOGITS, INPUT_EMBEDS, INPUT_IDS_NAME,
    LAST_HIDDEN_STATE, LOGITS, POSITION_IDS, START_LOGITS, TOKEN_TYPE_IDS,
};
use crate::pipelines::onnx::conversion::{ort_output_to_array_f32, ONNXInput};
use crate::RustBertError;
use ndarray::ArrayD;
use ort::session::Session;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// # ONNX Encoder model
/// Container for an ONNX encoder model and the corresponding session. Can be used individually for
/// pure-encoder models (e.g. BERT) or as part of encoder/decoder architectures.
pub struct ONNXEncoder {
    session: Mutex<Session>,
    name_mapping: InputOutputNameMapping,
}

impl ONNXEncoder {
    /// Create a new `ONNXEncoder`. Requires a pointer to the model file for
    /// the encoder and an ONNX environment configuration.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXEncoder;
    /// use std::path::PathBuf;
    /// let onnx_config = ONNXEnvironmentConfig::default();
    /// let model_file = PathBuf::from("path/to/model.onnx");
    ///
    /// let encoder = ONNXEncoder::new(model_file, &onnx_config).unwrap();
    /// ```
    pub fn new(
        model_file: PathBuf,
        onnx_config: &ONNXEnvironmentConfig,
    ) -> Result<Self, RustBertError> {
        let session = onnx_config
            .get_session_builder()?
            .commit_from_file(model_file)?;
        let name_mapping = get_input_output_mapping(&session);
        Ok(Self {
            session: Mutex::new(session),
            name_mapping,
        })
    }

    /// Forward pass through the model.
    ///
    /// The outputs provided by the model depend on the underlying ONNX model and are all marked as optional to support a broad range of
    /// encoder stacks for multiple stacks. The end-user should extract the required output that is provided by the model exported.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input array of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    ///
    /// # Returns
    ///
    /// * `ONNXEncoderModelOutput` containing:
    ///   - `last_hidden_state` - Optional array of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `logits` - Optional array of shape (*batch size*, *num_labels*)
    ///   - `start_logits` - Optional array of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - Optional array of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `hidden_states` - `Option<Vec<ArrayD<f32>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `attentions` - `Option<Vec<ArrayD<f32>>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    pub fn forward(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        attention_mask: Option<&ArrayD<i64>>,
        token_type_ids: Option<&ArrayD<i64>>,
        position_ids: Option<&ArrayD<i64>>,
        input_embeds: Option<&ArrayD<f32>>,
    ) -> Result<ONNXEncoderModelOutput, RustBertError> {
        let mut input_dict: HashMap<&str, ONNXInput> = HashMap::new();
        if let Some(input_ids) = input_ids {
            input_dict.insert(INPUT_IDS_NAME, ONNXInput::I64(input_ids.clone()));
        }
        if let Some(attention_mask) = attention_mask {
            input_dict.insert(ATTENTION_MASK_NAME, ONNXInput::I64(attention_mask.clone()));
        }
        if let Some(token_type_ids) = token_type_ids {
            input_dict.insert(TOKEN_TYPE_IDS, ONNXInput::I64(token_type_ids.clone()));
        }
        if let Some(position_ids) = position_ids {
            input_dict.insert(POSITION_IDS, ONNXInput::I64(position_ids.clone()));
        }
        if let Some(input_embeds) = input_embeds {
            input_dict.insert(INPUT_EMBEDS, ONNXInput::F32(input_embeds.clone()));
        }

        let mut input_values = Vec::with_capacity(self.name_mapping.input_names.len());
        for input_name in &self.name_mapping.input_names {
            let input = input_dict.remove(input_name.as_str()).ok_or_else(|| {
                RustBertError::OrtError(format!("{input_name} not found but expected by model."))
            })?;
            input_values.push((input_name.clone(), input.into_value()?));
        }

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(input_values)?;

        fn extract_output(
            outputs: &ort::session::SessionOutputs,
            name: &str,
        ) -> Result<Option<ArrayD<f32>>, RustBertError> {
            outputs.get(name).map(ort_output_to_array_f32).transpose()
        }

        let last_hidden_state = extract_output(&outputs, LAST_HIDDEN_STATE)?;
        let logits = extract_output(&outputs, LOGITS)?;
        let start_logits = extract_output(&outputs, START_LOGITS)?;
        let end_logits = extract_output(&outputs, END_LOGITS)?;

        let (hidden_states, attentions) = if self.name_mapping.output_names.len() > 1 {
            let hidden_states = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("hidden_states"))
                .map(|(name, _)| {
                    outputs
                        .get(name.as_str())
                        .ok_or_else(|| RustBertError::OrtError(format!("Output {name} not found.")))
                        .and_then(ort_output_to_array_f32)
                })
                .collect::<Result<Vec<_>, RustBertError>>()?;

            let attentions = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("attentions"))
                .map(|(name, _)| {
                    outputs
                        .get(name.as_str())
                        .ok_or_else(|| RustBertError::OrtError(format!("Output {name} not found.")))
                        .and_then(ort_output_to_array_f32)
                })
                .collect::<Result<Vec<_>, RustBertError>>()?;
            (Some(hidden_states), Some(attentions))
        } else {
            (None, None)
        };
        Ok(ONNXEncoderModelOutput {
            last_hidden_state,
            logits,
            start_logits,
            end_logits,
            hidden_states,
            attentions,
        })
    }
}

/// # ONNX encoder model output.
/// The outputs provided by the model depend on the underlying ONNX model and are all marked as optional to support a broad range of
/// encoder stacks for multiple stacks. The end-user should extract the required output that is provided by the model exported.
pub struct ONNXEncoderModelOutput {
    /// Last hidden states, typically used by masked language model encoder models
    pub last_hidden_state: Option<ArrayD<f32>>,
    /// logits, typically used by models with a sequence of classification head
    pub logits: Option<ArrayD<f32>>,
    /// logits marking the start location of a span (e.g. for extractive question answering tasks)
    pub start_logits: Option<ArrayD<f32>>,
    /// logits marking the end location of a span (e.g. for extractive question answering tasks)
    pub end_logits: Option<ArrayD<f32>>,
    /// Hidden states for intermediate layers of the model
    pub hidden_states: Option<Vec<ArrayD<f32>>>,
    /// Attention weights for intermediate layers of the model
    pub attentions: Option<Vec<ArrayD<f32>>>,
}
