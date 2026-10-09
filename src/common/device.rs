//! Backend-agnostic device selection.
//!
//! This enum decouples the pipelines configuration from the inference backend:
//! under the `libtorch` feature it can be converted to a `tch::Device`, and for
//! ONNX models it selects the execution providers used by the `ort` sessions.
//!
//! The `Mps` and `Vulkan` variants only exist under the `libtorch` feature
//! (the ONNX Runtime has no MPS/Vulkan execution providers, so an onnx-only
//! build never sees them). They round-trip losslessly to `tch::Device`, which
//! restores the upstream (pre-fork) ability to drive the libtorch backend on
//! Apple Silicon GPUs; requesting them on the ONNX path falls back to CPU
//! execution providers.

/// Device used to run the models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU device
    #[default]
    Cpu,
    /// CUDA device, with the associated device index
    Cuda(usize),
    /// Apple Silicon GPU (Metal Performance Shaders) — libtorch backend only
    #[cfg(feature = "libtorch")]
    Mps,
    /// Vulkan device — libtorch backend only
    #[cfg(feature = "libtorch")]
    Vulkan,
}

impl Device {
    /// Returns the CUDA device index if this is a CUDA device.
    pub fn cuda_device_id(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(device_id) => Some(*device_id),
            #[cfg(feature = "libtorch")]
            Device::Mps | Device::Vulkan => None,
        }
    }

    /// Returns a CUDA device if one is available, the CPU device otherwise.
    ///
    /// Note: this never returns `Mps`/`Vulkan` even where available — those
    /// backends are opt-in (construct `Device::Mps` explicitly), mirroring
    /// `tch::Device::cuda_if_available()`.
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
            tch::Device::Mps => Device::Mps,
            tch::Device::Vulkan => Device::Vulkan,
        }
    }
}

#[cfg(feature = "libtorch")]
impl From<Device> for tch::Device {
    fn from(device: Device) -> Self {
        match device {
            Device::Cpu => tch::Device::Cpu,
            Device::Cuda(device_id) => tch::Device::Cuda(device_id),
            Device::Mps => tch::Device::Mps,
            Device::Vulkan => tch::Device::Vulkan,
        }
    }
}

#[cfg(all(test, feature = "libtorch"))]
mod tests {
    use super::*;

    #[test]
    fn device_roundtrips_through_tch() {
        for device in [
            Device::Cpu,
            Device::Cuda(0),
            Device::Cuda(3),
            Device::Mps,
            Device::Vulkan,
        ] {
            let via_tch: tch::Device = device.into();
            let back: Device = via_tch.into();
            assert_eq!(device, back, "roundtrip lost information for {device:?}");
        }
    }

    #[test]
    fn mps_and_vulkan_survive_from_tch() {
        // Regression: the pre-Mps enum used to collapse these to Cpu, which
        // silently downgraded libtorch mac builds to CPU inference.
        assert_eq!(Device::from(tch::Device::Mps), Device::Mps);
        assert_eq!(Device::from(tch::Device::Vulkan), Device::Vulkan);
        let mps: tch::Device = Device::Mps.into();
        assert!(matches!(mps, tch::Device::Mps));
    }

    #[test]
    fn cuda_device_id_is_none_for_non_cuda() {
        assert_eq!(Device::Cuda(7).cuda_device_id(), Some(7));
        assert_eq!(Device::Cpu.cuda_device_id(), None);
        assert_eq!(Device::Mps.cuda_device_id(), None);
        assert_eq!(Device::Vulkan.cuda_device_id(), None);
    }
}
