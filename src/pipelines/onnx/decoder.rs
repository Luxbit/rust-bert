use crate::pipelines::generation_utils::{Cache, LMModelOutput};
use crate::pipelines::onnx::common::{get_input_output_mapping, InputOutputNameMapping};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, ENCODER_ATTENTION_MASK_NAME,
    ENCODER_HIDDEN_STATES_NAME, INPUT_IDS_NAME, POSITION_IDS,
};
use crate::pipelines::onnx::conversion::ONNXInput;
use crate::pipelines::onnx::models::ONNXLayerCache;
use crate::RustBertError;
use ndarray::ArrayD;
use ort::session::Session;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct ONNXDecoder {
    session: Mutex<Session>,
    name_mapping: InputOutputNameMapping,
    use_cache: bool,
}

impl ONNXDecoder {
    pub fn new(
        model_file: PathBuf,
        use_cache: bool,
        onnx_config: &ONNXEnvironmentConfig,
    ) -> Result<Self, RustBertError> {
        let session = onnx_config
            .get_session_builder()?
            .commit_from_file(model_file)?;
        let name_mapping = get_input_output_mapping(&session);
        Ok(Self {
            session: Mutex::new(session),
            name_mapping,
            use_cache,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&ArrayD<i64>>,
        attention_mask: Option<&ArrayD<i64>>,
        encoder_hidden_states: Option<&ArrayD<f32>>,
        encoder_attention_mask: Option<&ArrayD<i64>>,
        position_ids: Option<&ArrayD<i64>>,
        layer_states: Option<&ONNXLayerCache>,
    ) -> Result<LMModelOutput, RustBertError> {
        let mut input_dict: HashMap<&str, ONNXInput> = HashMap::new();
        if let Some(input_ids) = input_ids {
            input_dict.insert(INPUT_IDS_NAME, ONNXInput::I64(input_ids.clone()));
        }
        if let Some(attention_mask) = attention_mask {
            input_dict.insert(ATTENTION_MASK_NAME, ONNXInput::I64(attention_mask.clone()));
        }
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            input_dict.insert(
                ENCODER_HIDDEN_STATES_NAME,
                ONNXInput::F32(encoder_hidden_states.clone()),
            );
        }
        if let Some(encoder_attention_mask) = encoder_attention_mask {
            input_dict.insert(
                ENCODER_ATTENTION_MASK_NAME,
                ONNXInput::I64(encoder_attention_mask.clone()),
            );
        }
        if let Some(position_ids) = position_ids {
            input_dict.insert(POSITION_IDS, ONNXInput::I64(position_ids.clone()));
        }

        let mut input_values = Vec::with_capacity(self.name_mapping.input_names.len());
        for input_name in &self.name_mapping.input_names {
            if let Some(input) = input_dict.remove(input_name.as_str()) {
                input_values.push((input_name.clone(), input.into_value()?));
            } else {
                let layer_states = layer_states.ok_or_else(|| {
                    RustBertError::OrtError(format!(
                        "{input_name} not found and cache was not provided."
                    ))
                })?;
                let cached_state = layer_states
                    .values
                    .get(&input_name.replace("past", "present"))
                    .or_else(|| {
                        layer_states
                            .values
                            .get(&input_name.replace("past_key_values", "present"))
                    })
                    .ok_or_else(|| {
                        let found_keys = layer_states.values.keys().collect::<Vec<&String>>();
                        RustBertError::OrtError(format!(
                            "{input_name} not found in cache ({found_keys:?})."
                        ))
                    })?;
                input_values.push((
                    input_name.clone(),
                    ONNXInput::F32(cached_state.clone()).into_value()?,
                ));
            }
        }

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(input_values)?;

        let logits_value = outputs
            .get("logits")
            .ok_or_else(|| RustBertError::OrtError("Logits output not found.".to_string()))?;
        let lm_logits_array =
            crate::pipelines::onnx::conversion::ort_output_to_array_f32(logits_value)?;

        #[cfg(feature = "libtorch")]
        let lm_logits = crate::common::tensor_conversion::array_to_tensor_f32(&lm_logits_array)?;
        #[cfg(not(feature = "libtorch"))]
        let lm_logits = lm_logits_array;

        let cache = if self.use_cache {
            Cache::ONNXCache(ONNXLayerCache::from_ort_output(
                &outputs,
                &self.name_mapping.key_value_output_names,
            )?)
        } else {
            Cache::None
        };

        Ok(LMModelOutput { lm_logits, cache })
    }
}
