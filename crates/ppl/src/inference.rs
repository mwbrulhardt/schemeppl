//! MCMC Inference and Kernel System
//!
//! This module provides a flexible kernel-based system for MCMC inference,
//! similar to Gen.jl's kernel composition approach.

use crate::address::Selection;
use crate::choice_map::{ChoiceMap, ChoiceMapQuery};
use crate::gfi::{self, ArgDiff};
use rand::{rngs::StdRng, Rng};
use std::sync::{Arc, Mutex};

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
    R: Clone + std::fmt::Debug + 'static,
    V: Clone + std::fmt::Debug + 'static,
    A: Clone + std::fmt::Debug + AsRef<[E]> + 'static,
    T: gfi::Trace<R, V, Args = A> + Clone,
{
    let check = check.unwrap_or(false);
    let empty_observations = T::ChoiceMap::empty();
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
    R: Clone + std::fmt::Debug + 'static,
    V: Clone + std::fmt::Debug + 'static,
    A: Clone + std::fmt::Debug + AsRef<[E]> + 'static,
    T: gfi::Trace<R, V, Args = A> + Clone,
    G: gfi::GenerativeFunction<R, V, Args = A, TraceType = T>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::Address;
    use crate::dsl::ast::{Literal, Value};
    use crate::gfi::{GenerativeFunction, Trace};
    use crate::r#gen;
    use crate::utils::compute_mean_and_variance;
    use rand::distributions::Bernoulli;
    use rand::distributions::Distribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use statrs::distribution::Normal as RandNormal;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_gmm_with_dsl_proposal() {
        const SEED: u64 = 42;
        const BURN_IN: usize = 100;
        const DRAW: usize = 1000;
        const STEP_SIZE: f64 = 0.15;

        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(SEED)));

        let num_samples = 200;
        let mu1 = -2.0;
        let mu2 = 2.0;
        let sigma = 1.0;
        let p = 0.5;
        let z_dist = Bernoulli::new(p).unwrap();
        let z: Vec<bool> = (0..num_samples)
            .map(|_| z_dist.sample(&mut *rng.lock().unwrap()))
            .collect();

        let component1 = RandNormal::new(mu1, sigma).unwrap();
        let c1: Vec<f64> = (0..num_samples)
            .map(|_| component1.sample(&mut *rng.lock().unwrap()))
            .collect();

        let component2 = RandNormal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples)
            .map(|_| component2.sample(&mut *rng.lock().unwrap()))
            .collect();

        let data = Value::List(
            (0..num_samples)
                .map(|i| if z[i] { c1[i] } else { c2[i] })
                .into_iter()
                .map(|x| Value::Float(x))
                .collect(),
        );

        let model = gen!([data] {
            // Priors
            (sample mu1 (normal 0.0 1.0))
            (sample mu2 (normal 0.0 1.0))

            // Ordering
            (constrain (< mu1 mu2))

            // Mixture
            (define p 0.5)
            (define mix (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p))))

            (define observe-point (lambda (x) (observe (gensym) mix x)))

            (for-each observe-point data)
        });

        // Define a random-walk proposal using the DSL
        // This takes current values and step size as separate arguments
        let proposal = gen!([current_mu1, current_mu2, step_size] {
            // Propose new values using normal distributions centered at current values
            (sample mu1 (normal current_mu1 step_size))
            (sample mu2 (normal current_mu2 step_size))

            // Return a dummy value (proposals are about the choices, not the return value)
            #t
        });

        let mu1_name = "mu1".to_string();
        let mu2_name = "mu2".to_string();

        let program = model;
        let proposal_fn = proposal;
        let inference_rng = Arc::new(Mutex::new(StdRng::seed_from_u64(SEED)));

        let mut trace = program.simulate(inference_rng.clone(), vec![data.clone()]);
        let mut attempts = 0;

        while !trace.get_score().is_finite() && attempts < 1000 {
            trace = program.simulate(inference_rng.clone(), vec![data.clone()]);
            attempts += 1;
        }

        if !trace.get_score().is_finite() {
            panic!(
                "Unable to initialise Markov chain: failed to obtain finite posterior score after {} attempts. Check whether the chosen parameters (e.g. very tight σ or extreme μ) are compatible with the prior.",
                attempts
            );
        }

        // Burn-in
        for _ in 0..BURN_IN {
            // Get current values to pass to proposal
            let current_mu1 = trace
                .get_value(&Address::Symbol(mu1_name.clone()))
                .map(|record| record_to_value(&record))
                .unwrap_or(Value::Float(0.0));
            let current_mu2 = trace
                .get_value(&Address::Symbol(mu2_name.clone()))
                .map(|record| record_to_value(&record))
                .unwrap_or(Value::Float(0.0));

            let proposal_args = vec![current_mu1, current_mu2, Value::Float(STEP_SIZE)];

            let (new_trace, _accepted) = metropolis_hastings_with_proposal(
                inference_rng.clone(),
                trace,
                &proposal_fn,
                proposal_args,
                false,
                None,
            )
            .unwrap();
            trace = new_trace;
        }

        let mut history = Vec::with_capacity(DRAW);
        let mut num_accepted = 0u32;

        // Sampling
        for _ in 0..DRAW {
            // Get current values to pass to proposal
            let current_mu1 = trace
                .get_value(&Address::Symbol(mu1_name.clone()))
                .map(|record| record_to_value(&record))
                .unwrap_or(Value::Float(0.0));
            let current_mu2 = trace
                .get_value(&Address::Symbol(mu2_name.clone()))
                .map(|record| record_to_value(&record))
                .unwrap_or(Value::Float(0.0));

            let proposal_args = vec![
                current_mu1.clone(),
                current_mu2.clone(),
                Value::Float(STEP_SIZE),
            ];

            let (new_trace, accepted) = metropolis_hastings_with_proposal(
                inference_rng.clone(),
                trace,
                &proposal_fn,
                proposal_args,
                false,
                None,
            )
            .unwrap();
            history.push(new_trace.clone());

            trace = new_trace;

            if accepted {
                num_accepted += 1;
            }
        }

        let (mean_mu1, variance_mu1) = compute_mean_and_variance(&history, &mu1_name);
        let (mean_mu2, variance_mu2) = compute_mean_and_variance(&history, &mu2_name);

        println!("DSL Proposal - mean_mu1: {}", mean_mu1);
        println!("DSL Proposal - variance_mu1: {}", variance_mu1);

        println!("DSL Proposal - mean_mu2: {}", mean_mu2);
        println!("DSL Proposal - variance_mu2: {}", variance_mu2);

        let acceptance_rate = num_accepted as f64 / DRAW as f64;
        println!("DSL Proposal - acceptance_rate: {}", acceptance_rate);
        assert!(acceptance_rate > 0.1 && acceptance_rate < 0.9); // Slightly wider range for DSL version

        assert!((mean_mu1 - mu1).abs() < 0.6); // Slightly more tolerance
        assert!(variance_mu1 > 0.0 && variance_mu1 < 2.5);

        assert!((mean_mu2 - mu2).abs() < 0.6);
        assert!(variance_mu2 > 0.0 && variance_mu2 < 2.5);
    }

    // Helper to convert Record to Value
    fn record_to_value(record: &crate::dsl::trace::Record) -> Value {
        match record {
            crate::dsl::trace::Record::Choice(literal, _) => literal_to_value(literal),
            crate::dsl::trace::Record::Call(_, _) => Value::String("call".to_string()),
        }
    }

    // The trace get_value method now returns a choice value instead of Value
    // So we need to convert to Value and extract the expected type
    fn literal_to_value(literal: &Literal) -> Value {
        match literal {
            Literal::Boolean(b) => Value::Boolean(*b),
            Literal::Integer(i) => Value::Integer(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
        }
    }
}
