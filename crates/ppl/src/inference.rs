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

/// A function type that takes in some partial arguments, and a trace, 
/// and returns the full arguments that can be used to input into a proposal function.
pub type ExtractArgs<Args, T> = Box<dyn Fn(&Args, &T) -> Result<Args, String>>;

/// Check that observed choices in the new trace match the expected observations
fn check_observations<X>(_choices: &X, _observations: &X) -> Result<(), String>
where
    X: Clone + Debug + Index<Address> + 'static,
{
    // For now, skip detailed checking since this is just a placeholder
    // In a full implementation, this would check that all observed values match
    // TODO: Augment the definition of an abstract choice map such that this check is possible 
    //       without importing SchemeChoiceMap
    Ok(())
}

/// Generic Metropolis-Hastings update using ancestral sampling (regenerate)
pub fn metropolis_hastings<X, R, T, G>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    gen_fn: &G,
    selection: Selection,
    check: bool,
    observations: X,
) -> Result<(T, bool), String>
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
    T: Trace<X, R> + Clone,
    T::Args: Clone,
    G: GenerativeFunction<X, R, TraceType = T, Args = T::Args>,
{
    let model_args = trace.get_args().clone();

    let (new_trace, weight, _) = gen_fn
        .regenerate(rng.clone(), trace.clone(), selection, model_args)
        .map_err(|e| format!("Regenerate failed: {:?}", e))?;

    if check {
        let new_choices = new_trace.get_choices();
        check_observations(&new_choices, &observations)?;
    }

    // Acceptance criterion
    let u: f64 = rng.lock().unwrap().gen();
    let alpha = weight.iter().sum::<f64>();

    if u < alpha.exp() {
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
    extract_args: &ExtractArgs<G::Args, T>,
    check: bool,
    observations: X,
) -> Result<(T, bool), String>
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
    T: Trace<X, R> + Clone,
    T::Args: Clone,
    G: GenerativeFunction<X, R, TraceType = T>,
    G::Args: Clone + Debug,
{
    let model_args = trace.get_args().clone();

    // Forward proposal - propose new choices and get the proposal density
    let proposal_args_forward = extract_args(&proposal_args, &trace)?;
    let (fwd_choices, fwd_weight_array, _marker) = proposal
        .propose(rng.clone(), proposal_args_forward.clone())
        .map_err(|e| format!("Forward proposal failed: {:?}", e))?;
    let fwd_weight = fwd_weight_array.iter().sum::<f64>();

    // Update the trace with the proposed choices
    let (new_trace, weight_array, discard) = trace
        .update(rng.clone(), fwd_choices, model_args)
        .map_err(|e| format!("Update failed: {:?}", e))?;
    let weight = weight_array.iter().sum::<f64>();

    // Backward proposal - assess the discarded choices
    let proposal_args_backward = extract_args(&proposal_args, &new_trace)?;
    let (bwd_density, _retval): (Density, R) = proposal
        .assess(rng.clone(), proposal_args_backward, discard)
        .map_err(|e| format!("Backward proposal failed: {:?}", e))?;
    let bwd_weight = bwd_density.iter().sum::<f64>();

    if check {
        check_observations(&new_trace.get_choices(), &observations)?;
    }

    // Acceptance criterion
    let u: f64 = rng.lock().unwrap().gen();
    let alpha = weight - fwd_weight + bwd_weight;

    if u < alpha.exp() {
        Ok((new_trace, true))
    } else {
        Ok((trace.clone(), false))
    }
}
