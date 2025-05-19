use crate::eval::GenerativeFunction;
use crate::trace::Trace;
use crate::Value;

/// Compute the mean and variance of a given address in the history
pub fn compute_mean_and_variance(
    history: &Vec<Trace<GenerativeFunction, String, Value>>,
    addr: &str,
) -> (f64, f64) {
    let mean = history
        .iter()
        .map(|t| t.get_choice(&addr.to_string()).value.expect_float())
        .sum::<f64>()
        / history.len() as f64;
    let variance = history
        .iter()
        .map(|t| (t.get_choice(&addr.to_string()).value.expect_float() - mean).powi(2))
        .sum::<f64>()
        / history.len() as f64;
    (mean, variance)
}
