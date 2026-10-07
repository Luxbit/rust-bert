use serde::{Deserialize, Serialize};
#[cfg(feature = "libtorch")]
use std::f64::consts::PI;
#[cfg(feature = "libtorch")]
use tch::Tensor;

#[cfg(feature = "libtorch")]
pub fn _gelu(x: &Tensor) -> Tensor {
    x * 0.5 * (1.0 + (x / ((2.0_f64).sqrt())).erf())
}

#[cfg(feature = "libtorch")]
pub fn _relu(x: &Tensor) -> Tensor {
    x.relu()
}

#[cfg(feature = "libtorch")]
pub fn _swish(x: &Tensor) -> Tensor {
    x * x.sigmoid()
}

#[cfg(feature = "libtorch")]
pub fn _mish(x: &Tensor) -> Tensor {
    x * (x.softplus().tanh())
}

#[cfg(feature = "libtorch")]
pub fn _gelu_new(x: &Tensor) -> Tensor {
    x * 0.5 * (((x.pow_tensor_scalar(3.0f64) * 0.044715 + x) * ((2f64 / PI).sqrt())).tanh() + 1)
}

#[cfg(feature = "libtorch")]
pub fn _tanh(x: &Tensor) -> Tensor {
    x.tanh()
}

#[cfg(feature = "libtorch")]
pub fn _identity(x: &Tensor) -> Tensor {
    x.shallow_clone()
}

#[cfg(feature = "libtorch")]
pub struct TensorFunction(Box<fn(&Tensor) -> Tensor>);

#[cfg(feature = "libtorch")]
impl TensorFunction {
    pub fn new(fun: Box<fn(&Tensor) -> Tensor>) -> Self {
        Self(fun)
    }

    pub fn get_fn(&self) -> &fn(&Tensor) -> Tensor {
        &self.0
    }
}
#[cfg(feature = "libtorch")]
impl std::fmt::Debug for TensorFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "TensorFunction")
    }
}
#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish ([Ramachandran, 2017](https://arxiv.org/abs/1710.05941))
    swish,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08691))
    mish,
    /// Gaussian Error Linear Unit (New) ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Tanh
    tanh,
    /// Identity
    identity,
}

impl Activation {
    #[cfg(feature = "libtorch")]
    pub fn get_function(&self) -> TensorFunction {
        TensorFunction::new(Box::new(match self {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::swish => _swish,
            Activation::gelu_new => _gelu_new,
            Activation::mish => _mish,
            Activation::tanh => _tanh,
            Activation::identity => _identity,
        }))
    }
}

#[cfg(all(test, feature = "libtorch"))]
mod test {
    use super::*;
    #[test]
    #[ignore]
    fn tensorfunction_send() {
        let _: Box<dyn Send> = Box::new(Activation::gelu.get_function());
    }
}
