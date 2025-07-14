use num_traits::ToPrimitive;

use crate::address::Address;
use crate::dsl::trace::SchemeDSLTrace;

/// Compute the mean and variance of a given address in the history
/// Works specifically with SchemeDSLTrace and returns (mean, variance) for float values
pub fn compute_mean_and_variance(history: &Vec<SchemeDSLTrace>, addr: &str) -> (f64, f64) {
    let address = Address::Symbol(addr.to_string());
    let values: Vec<f64> = history
        .iter()
        .filter_map(|trace| {
            trace
                .get_choice_value(&address)
                .and_then(|value| value.to_f64())
        })
        .collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance)
}
