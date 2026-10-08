use crate::RustBertError;
use tch::{Kind, Scalar};

pub(crate) fn get_negative_infinity(kind: Kind) -> Result<Scalar, RustBertError> {
    Ok(match kind {
        Kind::Uint8 => Scalar::int(u8::MIN.into()),
        Kind::Int8 => Scalar::int(i8::MIN.into()),
        Kind::Int16 => Scalar::int(i16::MIN.into()),
        Kind::Int => Scalar::int(i32::MIN.into()),
        Kind::Int64 => Scalar::int(i64::MIN),
        Kind::Half => Scalar::float(half::f16::NEG_INFINITY.into()),
        Kind::Float => Scalar::float(f32::NEG_INFINITY.into()),
        Kind::BFloat16 => Scalar::float(half::bf16::NEG_INFINITY.into()),
        Kind::Double => Scalar::float(f64::NEG_INFINITY),
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get negative infinity for {kind:?}",
            )))
        }
    })
}

pub(crate) fn get_min(kind: Kind) -> Result<Scalar, RustBertError> {
    Ok(match kind {
        Kind::Uint8 => Scalar::int(u8::MIN.into()),
        Kind::Int8 => Scalar::int(i8::MIN.into()),
        Kind::Int16 => Scalar::int(i16::MIN.into()),
        Kind::Int => Scalar::int(i32::MIN.into()),
        Kind::Int64 => Scalar::int(i64::MIN),
        Kind::Half => Scalar::float(half::f16::MIN.into()),
        Kind::Float => Scalar::float(f32::MIN.into()),
        Kind::BFloat16 => Scalar::float(half::bf16::MIN.into()),
        Kind::Double => Scalar::float(f64::MIN),
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get min for {kind:?}",
            )))
        }
    })
}
