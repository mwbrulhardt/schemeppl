//! MCMC Inference and Kernel System
//!
//! This module provides a flexible kernel-based system for MCMC inference,
//! similar to Gen.jl's kernel composition approach.

use rand::{rngs::StdRng, Rng};
use std::sync::{Arc, Mutex};

use crate::ast::{Value, Literal};
use crate::core::gfi::{self, ArgDiff};
use crate::core::address::{Selection, ChoiceMap};


/// Check that observed choices in the new trace match the expected observations
fn check_observations(
    _choices: &ChoiceMap<Literal>,
    _observations: &ChoiceMap<Literal>,
) -> Result<(), String> {
    // For now, skip detailed checking since this is just a placeholder
    // In a full implementation, this would check that all observed values match
    Ok(())
}

/// Generic Metropolis-Hastings update using ancestral sampling (regenerate)
pub fn metropolis_hastings<T>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    selection: Selection,
    check: Option<bool>,
    observations: Option<ChoiceMap<Literal>>,
) -> Result<(T, bool), String>
where
    T: gfi::Trace<Literal, Args = Vec<Value>, RetVal = Value> + Clone,
{
    let check = check.unwrap_or(false);
    let empty_observations = ChoiceMap::Empty;
    let observations = observations.unwrap_or(empty_observations);

    let model_args = trace.get_args().clone();
    let argdiffs = vec![ArgDiff::NoChange; model_args.len()];

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
pub fn metropolis_hastings_with_proposal<T, G>(
    rng: Arc<Mutex<StdRng>>,
    trace: T,
    proposal: &G,
    proposal_args: Vec<Value>,
    check: bool,
    observations: Option<ChoiceMap<Literal>>,
) -> Result<(T, bool), String>
where
    T: gfi::Trace<Literal, Args = Vec<Value>, RetVal = Value> + Clone,
    G: gfi::GenerativeFunction<Literal, Args = Vec<Value>, RetVal = Value, TraceType = T>,
{
    let model_args = trace.get_args().clone();
    let argdiffs = vec![ArgDiff::NoChange; model_args.len()];

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
    
    if u < alpha.exp() {
        Ok((new_trace, true))
    } else {
        Ok((trace.clone(), false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#gen;
    use crate::utils::compute_mean_and_variance;
    use rand::distributions::Bernoulli;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use statrs::distribution::Normal as RandNormal;
    use std::sync::{Arc, Mutex};
    use crate::core::gfi::{GenerativeFunction, Trace};
    use rand::distributions::Distribution;
    use crate::core::address::Address;


    #[test]
    fn test_gmm_with_dsl_proposal() {
        const SEED: u64 = 42;
        const BURN_IN: usize = 100;
        const DRAW: usize = 1000;
        const STEP_SIZE: f64 = 0.15;

        let data_seed = 40;
        let mut rng = StdRng::seed_from_u64(data_seed);

        let num_samples = 100;
        let mu1 = -2.0;
        let mu2 = 2.0;
        let sigma = 1.0;
        let p = 0.5;
        let z_dist = Bernoulli::new(p).unwrap();
        let z: Vec<bool> = (0..num_samples).map(|_| z_dist.sample(&mut rng)).collect();

        let component1 = RandNormal::new(mu1, sigma).unwrap();
        let c1: Vec<f64> = (0..num_samples)
            .map(|_| component1.sample(&mut rng))
            .collect();

        let component2 = RandNormal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples)
            .map(|_| component2.sample(&mut rng))
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

        let mut trace = program.simulate(inference_rng.clone(), vec![data]);
        
        // Burn-in
        for _ in 0..BURN_IN {
            // Get current values to pass to proposal
            let current_mu1 = trace.get_value(&Address::Symbol(mu1_name.clone()))
                .map(|literal| literal_to_value(&literal))
                .unwrap_or(Value::Float(0.0));
            let current_mu2 = trace.get_value(&Address::Symbol(mu2_name.clone()))
                .map(|literal| literal_to_value(&literal))
                .unwrap_or(Value::Float(0.0));
            
            let proposal_args = vec![
                current_mu1,
                current_mu2,
                Value::Float(STEP_SIZE)
            ];
            
            let (new_trace, _accepted) = metropolis_hastings_with_proposal(
                inference_rng.clone(),
                trace,
                &proposal_fn,
                proposal_args,
                false,
                None,
            ).unwrap();
            trace = new_trace;
        }

        let mut history = Vec::with_capacity(DRAW);
        let mut num_accepted = 0u32;

        // Sampling
        for _ in 0..DRAW {
            // Get current values to pass to proposal
            let current_mu1 = trace.get_value(&Address::Symbol(mu1_name.clone()))
                .map(|literal| literal_to_value(&literal))
                .unwrap_or(Value::Float(0.0));
            let current_mu2 = trace.get_value(&Address::Symbol(mu2_name.clone()))
                .map(|literal| literal_to_value(&literal))
                .unwrap_or(Value::Float(0.0));
            
            let proposal_args = vec![
                current_mu1,
                current_mu2,
                Value::Float(STEP_SIZE)
            ];
            
            let (new_trace, accepted) = metropolis_hastings_with_proposal(
                inference_rng.clone(),
                trace,
                &proposal_fn,
                proposal_args,
                false,
                None,
            ).unwrap();
            trace = new_trace;
            
            if accepted {
                num_accepted += 1;
            }
            history.push(trace.clone());
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

    // The trace get_value method now returns Literal instead of Value
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


