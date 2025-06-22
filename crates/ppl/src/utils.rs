use num_traits::ToPrimitive;
use std::fmt::Debug;

use crate::address::Address;
use crate::gfi::Trace;

/// Compute the mean and variance of a given address in the history
pub fn compute_mean_and_variance<R, V, T>(history: &Vec<T>, addr: &str) -> (f64, f64)
where
    R: Clone + Debug + 'static,
    V: Clone + Debug + ToPrimitive + 'static,
    T: Trace<R, V>,
{
    let address = Address::Symbol(addr.to_string());
    let values: Vec<f64> = history
        .iter()
        .filter_map(|t| t.get_value(&address).and_then(|record| record.to_f64()))
        .collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance)
}
