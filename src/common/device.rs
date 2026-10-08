//! Backend-agnostic device selection.
//!
//! This enum decouples the pipelines configuration from the inference backend:
//! under the `libtorch` feature it can be converted to a `tch::Device`, and for
//! ONNX models it selects the execution providers used by the `ort` sessions.

/// Device used to run the models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU device
    #[default]
    Cpu,
    /// CUDA device, with the associated device index
    Cuda(usize),
}

impl Device {
    /// Returns the CUDA device index if this is a CUDA device.
    pub fn cuda_device_id(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(device_id) => Some(*device_id),
        }
    }

    /// Returns a CUDA device if one is available, the CPU device otherwise.
    pub fn cuda_if_available() -> Self {
        #[cfg(feature = "cuda")]
        {
            use ort::ep::ExecutionProvider;
            if ort::ep::CUDA::default().is_available().unwrap_or(false) {
                return Device::Cuda(0);
            }
        }
        #[cfg(all(feature = "libtorch", not(feature = "cuda")))]
        return Device::from(tch::Device::cuda_if_available());
        #[cfg(any(not(feature = "libtorch"), feature = "cuda"))]
        Device::Cpu
    }
}

#[cfg(feature = "libtorch")]
impl From<tch::Device> for Device {
    fn from(device: tch::Device) -> Self {
        match device {
            tch::Device::Cpu => Device::Cpu,
            tch::Device::Cuda(device_id) => Device::Cuda(device_id),
            // MPS / Vulkan have no equivalent in the ONNX Runtime path; fall back to CPU
            tch::Device::Mps | tch::Device::Vulkan => Device::Cpu,
        }
    }
}

#[cfg(feature = "libtorch")]
impl From<Device> for tch::Device {
    fn from(device: Device) -> Self {
        match device {
            Device::Cpu => tch::Device::Cpu,
            Device::Cuda(device_id) => tch::Device::Cuda(device_id),
        }
    }
}
