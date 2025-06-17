use crate::core::gfi::Trace;
use crate::core::address::Address;
use crate::ast::Literal;

/// Convert a Literal to f64 for numerical computations
fn literal_to_f64(literal: &Literal) -> Result<f64, String> {
    match literal {
        Literal::Float(f) => Ok(*f),
        Literal::Integer(i) => Ok(*i as f64),
        _ => Err(format!("Cannot convert literal {:?} to f64", literal)),
    }
}

/// Compute the mean and variance of a given address in the history
pub fn compute_mean_and_variance<T: Trace<Literal>>(
    history: &Vec<T>,
    addr: &str,
) -> (f64, f64) {
    let address = Address::Symbol(addr.to_string());
    let values: Vec<f64> = history
        .iter()
        .filter_map(|t| t.get_value(&address).and_then(|literal| literal_to_f64(&literal).ok()))
        .collect();
    
    if values.is_empty() {
        return (0.0, 0.0);
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance)
}
