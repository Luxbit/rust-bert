/// # Configuration for ONNX sessions
use crate::RustBertError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::{ep, ep::ExecutionProviderDispatch};

pub(crate) static INPUT_IDS_NAME: &str = "input_ids";
pub(crate) static ATTENTION_MASK_NAME: &str = "attention_mask";
pub(crate) static ENCODER_HIDDEN_STATES_NAME: &str = "encoder_hidden_states";
pub(crate) static ENCODER_ATTENTION_MASK_NAME: &str = "encoder_attention_mask";
pub(crate) static TOKEN_TYPE_IDS: &str = "token_type_ids";
pub(crate) static POSITION_IDS: &str = "position_ids";
pub(crate) static INPUT_EMBEDS: &str = "input_embeds";
pub(crate) static LAST_HIDDEN_STATE: &str = "last_hidden_state";
pub(crate) static LOGITS: &str = "logits";
pub(crate) static START_LOGITS: &str = "start_logits";
pub(crate) static END_LOGITS: &str = "end_logits";

#[derive(Default)]
/// # ONNX environment configuration
/// See <https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions>
pub struct ONNXEnvironmentConfig {
    pub optimization_level: Option<GraphOptimizationLevel>,
    pub execution_providers: Option<Vec<ExecutionProviderDispatch>>,
    pub num_intra_threads: Option<usize>,
    pub num_inter_threads: Option<usize>,
    pub parallel_execution: Option<bool>,
    pub enable_memory_pattern: Option<bool>,
}

impl ONNXEnvironmentConfig {
    /// Create a new `ONNXEnvironmentConfig` from a `rust_bert::Device`.
    /// This helper function maps the device to the ONNX Runtime execution providers.
    ///
    /// Note that using a CUDA device requires the `cuda` feature of this crate (which enables
    /// the `cuda` feature of the `ort` dependency and links the CUDA execution provider).
    /// Without this feature, CUDA devices fall back to CPU execution.
    ///
    /// `Device::Mps` / `Device::Vulkan` (libtorch-only variants) have no ONNX
    /// Runtime execution provider: they resolve to CPU execution here. Use
    /// them with the libtorch backend, not with ONNX sessions.
    pub fn from_device(device: crate::Device) -> Self {
        let mut execution_providers = Vec::new();
        if let Some(device_id) = device.cuda_device_id() {
            #[cfg(feature = "cuda")]
            execution_providers.push(ep::CUDA::default().with_device_id(device_id as i32).build());
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_id;
            }
        };
        execution_providers.push(ep::CPU::default().build());
        ONNXEnvironmentConfig {
            execution_providers: Some(execution_providers),
            ..Default::default()
        }
    }

    ///Build a session builder from an `ONNXEnvironmentConfig`.
    pub fn get_session_builder(&self) -> Result<SessionBuilder, RustBertError> {
        let mut session_builder = Session::builder()?;
        if let Some(optimization_level) = self.optimization_level {
            session_builder = session_builder.with_optimization_level(optimization_level)?;
        }
        if let Some(num_intra_threads) = self.num_intra_threads {
            session_builder = session_builder.with_intra_threads(num_intra_threads)?;
        }
        if let Some(num_inter_threads) = self.num_inter_threads {
            session_builder = session_builder.with_inter_threads(num_inter_threads)?;
        }
        if let Some(parallel_execution) = self.parallel_execution {
            session_builder = session_builder.with_parallel_execution(parallel_execution)?;
        }
        if let Some(enable_memory_pattern) = self.enable_memory_pattern {
            session_builder = session_builder.with_memory_pattern(enable_memory_pattern)?;
        }
        if let Some(execution_providers) = &self.execution_providers {
            session_builder = session_builder.with_execution_providers(execution_providers)?;
        }
        Ok(session_builder)
    }
}
