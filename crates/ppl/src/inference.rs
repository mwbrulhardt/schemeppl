//! MCMC Inference and Kernel System
//!
//! This module provides a flexible kernel-based system for MCMC inference,
//! similar to Gen.jl's kernel composition approach.
//!

use std::fmt::Debug;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use rand::{rngs::StdRng, Rng};

use crate::address::{Address, Selection};
use crate::gfi::{Density, GenerativeFunction, Trace};

/// Check that observed choices in the new trace match the expected observations
fn check_observations<X>(_choices: &X, _observations: &X) -> Result<(), String>
where
    X: Clone + Debug + Index<Address> + 'static,
{
    // For now, skip detailed checking since this is just a placeholder
    // In a full implementation, this would check that all observed values match
    Ok(())
}

/// Generic Metropolis-Hastings update using ancestral sampling (regenerate)
pub fn metropolis_hastings<X, R, T, G>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    gen_fn: &G,
    selection: Selection,
    check: Option<bool>,
    observations: Option<X>,
) -> Result<(T, bool), String>
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
    T: Trace<X, R> + Clone,
    T::Args: Clone,
    G: GenerativeFunction<X, R, TraceType = T, Args = T::Args>,
{
    let check = check.unwrap_or(false);
    let observations = observations.unwrap_or_else(|| {
        // Create an empty choice map - this depends on the specific implementation
        // For now, we'll skip this and handle it in the check
        panic!("Need to provide observations or implement default choice map creation")
    });

    let model_args = trace.get_args().clone();

    let (new_trace, weight, _) = gen_fn
        .regenerate(rng.clone(), trace.clone(), selection, model_args)
        .map_err(|e| format!("Regenerate failed: {:?}", e))?;

    let log_alpha = weight.iter().sum::<f64>(); // Sum the weight array

    if check {
        let new_choices = new_trace.get_choices();
        check_observations(&new_choices, &observations)?;
    }

    // Accept with probability exp(log_alpha)
    let mut rng_guard = rng.lock().unwrap();
    let u: f64 = rng_guard.gen();

    if u < log_alpha.exp() {
        Ok((new_trace, true))
    } else {
        Ok((trace, false))
    }
}

/// Generic MH with custom proposal generative function
pub fn metropolis_hastings_with_proposal<X, R, T, G>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    proposal: &G,
    proposal_args: G::Args,
    check: bool,
    observations: Option<X>,
) -> Result<(T, bool), String>
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
    T: Trace<X, R> + Clone,
    T::Args: Clone,
    G: GenerativeFunction<X, R, TraceType = T>,
    G::Args: Clone,
{
    let model_args = trace.get_args().clone();

    // Forward proposal - generate constraints using the proposal function
    let (proposal_trace, fwd_weight_array) = proposal
        .generate(rng.clone(), None, proposal_args.clone())
        .map_err(|e| format!("Forward proposal failed: {:?}", e))?;

    let fwd_choices = proposal_trace.get_choices();
    let fwd_weight = fwd_weight_array.iter().sum::<f64>();

    // Update the trace with the proposed choices
    let (new_trace, update_weight, discard) = trace
        .update(rng.clone(), fwd_choices, model_args)
        .map_err(|e| format!("Update failed: {:?}", e))?;

    let update_weight_sum = update_weight.iter().sum::<f64>();

    // Backward proposal - assess the discarded choices
    let (bwd_density, _retval): (Density, R) = if let Some(discarded) = discard {
        proposal
            .assess(rng.clone(), proposal_args, discarded)
            .map_err(|e| format!("Backward proposal failed: {:?}", e))?
    } else {
        // No choices were discarded, so backward probability is 1 (log prob = 0)
        let empty_density = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 0.0);
        let retval = proposal_trace.get_retval();
        (empty_density, retval)
    };

    let bwd_weight = bwd_density.iter().sum::<f64>();

    let log_alpha = update_weight_sum - fwd_weight + bwd_weight;

    if check {
        if let Some(ref obs) = observations {
            check_observations(&new_trace.get_choices(), obs)?;
        }
    }

    // Metropolis-Hastings acceptance criterion
    let u: f64 = rng.lock().unwrap().gen();

    if u < log_alpha.exp() {
        Ok((new_trace, true))
    } else {
        Ok((trace.clone(), false))
    }
}
