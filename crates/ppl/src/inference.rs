//! MCMC Inference and Kernel System
//!
//! This module provides a flexible kernel-based system for MCMC inference,
//! similar to Gen.jl's kernel composition approach.
//!

use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use rand::{rngs::StdRng, Rng};

use crate::address::Selection;
use crate::choice_map::{ChoiceMap, ChoiceMapQuery};
use crate::gfi::{ArgDiff, GenerativeFunction, Trace};

/// Check that observed choices in the new trace match the expected observations
fn check_observations<V, C: ChoiceMap<V>>(_choices: &C, _observations: &C) -> Result<(), String> {
    // For now, skip detailed checking since this is just a placeholder
    // In a full implementation, this would check that all observed values match
    Ok(())
}

/// Generic Metropolis-Hastings update using ancestral sampling (regenerate)
pub fn metropolis_hastings<R, V, T, A, E>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    selection: Selection,
    check: Option<bool>,
    observations: Option<T::ChoiceMap>,
) -> Result<(T, bool), String>
where
    R: Clone + Debug + 'static,
    V: Clone + Debug + 'static,
    A: Clone + Debug + AsRef<[E]> + 'static,
    T: Trace<R, V, Args = A> + Clone,
{
    let check = check.unwrap_or(false);
    let empty_observations = T::ChoiceMap::default();
    let observations = observations.unwrap_or(empty_observations);

    let model_args = trace.get_args().clone();
    let argdiffs = vec![ArgDiff::NoChange; model_args.as_ref().len()];

    let (new_trace, weight, _) = trace
        .regenerate(rng.clone(), model_args, argdiffs, selection)
        .map_err(|e| format!("Regenerate failed: {:?}", e))?;

    let alpha = weight;

    if check {
        let new_choices = new_trace.get_choices();
        check_observations(&new_choices, &observations)?;
    }

    // Accept with probability exp(log_alpha)
    let mut rng_guard = rng.lock().unwrap();
    let u: f64 = rng_guard.gen();

    if u < alpha.exp() {
        Ok((new_trace, true))
    } else {
        Ok((trace.clone(), false))
    }
}

/// Generic MH with custom proposal generative function
pub fn metropolis_hastings_with_proposal<R, V, T, G, A, E>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    proposal: &G,
    proposal_args: A,
    check: bool,
    observations: Option<T::ChoiceMap>,
) -> Result<(T, bool), String>
where
    R: Clone + Debug + 'static,
    V: Clone + Debug + 'static,
    A: Clone + Debug + AsRef<[E]> + 'static,
    T: Trace<R, V, Args = A> + Clone,
    G: GenerativeFunction<R, V, Args = A, TraceType = T>,
    T::ChoiceMap: ChoiceMapQuery<V>,
{
    let model_args = trace.get_args().clone();
    let argdiffs = vec![ArgDiff::NoChange; model_args.as_ref().len()];

    // Forward proposal
    let (fwd_choices, fwd_weight, _) = proposal
        .propose(rng.clone(), proposal_args.clone())
        .map_err(|e| format!("Forward proposal failed: {:?}", e))?;

    let (new_trace, weight, _, discard) = trace
        .update(rng.clone(), model_args, argdiffs.clone(), fwd_choices)
        .map_err(|e| format!("Update failed: {:?}", e))?;

    // Backward proposal
    let (bwd_weight, _) = proposal
        .assess(rng.clone(), proposal_args, discard)
        .map_err(|e| format!("Backward proposal failed: {:?}", e))?;

    let alpha = weight - fwd_weight + bwd_weight;

    if check {
        let new_choices = new_trace.get_choices();
        if let Some(ref obs) = observations {
            check_observations(&new_choices, obs)?;
        }
    }

    // Metropolis-Hastings acceptance criterion
    // Accept with probability exp(log_alpha)
    let mut rng_guard = rng.lock().unwrap();
    let u: f64 = rng_guard.gen();

    let accept = u < alpha.exp();

    if accept {
        Ok((new_trace, true))
    } else {
        Ok((trace.clone(), false))
    }
}
