use std::fmt::Debug;

use crate::address::Address;
use crate::gfi::Trace;

/// Convert a choice value to f64 for numerical computations
/// This trait allows different choice value types to be converted to f64
pub trait ToF64 {
    fn to_f64(&self) -> Result<f64, String>;
}

/// Compute the mean and variance of a given address in the history
pub fn compute_mean_and_variance<R, V, T>(history: &Vec<T>, addr: &str) -> (f64, f64)
where
    R: Clone + Debug + 'static,
    V: Clone + Debug + ToF64 + 'static,
    T: Trace<R, V>,
{
    let address = Address::Symbol(addr.to_string());
    let values: Vec<f64> = history
        .iter()
        .filter_map(|t| {
            t.get_value(&address)
                .and_then(|choice| choice.to_f64().ok())
        })
        .collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance)
}

// Implementations for common choice value types
impl ToF64 for crate::dsl::ast::Literal {
    fn to_f64(&self) -> Result<f64, String> {
        match self {
            crate::dsl::ast::Literal::Float(f) => Ok(*f),
            crate::dsl::ast::Literal::Integer(i) => Ok(*i as f64),
            _ => Err(format!("Cannot convert literal {:?} to f64", self)),
        }
    }
}

impl ToF64 for f64 {
    fn to_f64(&self) -> Result<f64, String> {
        Ok(*self)
    }
}

impl ToF64 for i64 {
    fn to_f64(&self) -> Result<f64, String> {
        Ok(*self as f64)
    }
}

impl ToF64 for i32 {
    fn to_f64(&self) -> Result<f64, String> {
        Ok(*self as f64)
    }
}
