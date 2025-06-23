use rand::distributions::Bernoulli;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::Normal as RandNormal;
use std::sync::{Arc, Mutex};

use ppl::dsl::Value;
use ppl::inference::metropolis_hastings_with_proposal;
use ppl::utils::compute_mean_and_variance;
use ppl::{r#gen, GenerativeFunction, Trace};

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

        (list mu1 mu2)
    });

    // Define a random-walk proposal using the DSL
    // This takes current values and step size as separate arguments
    let proposal = gen!([current_mu1, current_mu2, step_size] {
        // Propose new values using normal distributions centered at current values
        (sample mu1 (normal current_mu1 step_size))
        (sample mu2 (normal current_mu2 step_size))

        (list mu1 mu2)
    });

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
        // Get current values from the return value instead of individual trace values
        let (mu1, mu2) = match trace.get_retval() {
            Value::List(ref values) if values.len() >= 2 => (values[0].clone(), values[1].clone()),
            _ => (Value::Float(0.0), Value::Float(0.0)),
        };

        let proposal_args = vec![mu1, mu2, Value::Float(STEP_SIZE)];

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
        // Get current values from the return value instead of individual trace values
        let (mu1, mu2) = match trace.get_retval() {
            Value::List(ref values) if values.len() >= 2 => (values[0].clone(), values[1].clone()),
            _ => (Value::Float(0.0), Value::Float(0.0)),
        };

        let proposal_args = vec![mu1, mu2, Value::Float(STEP_SIZE)];

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

    let (mean_mu1, variance_mu1) = compute_mean_and_variance(&history, "mu1");
    let (mean_mu2, variance_mu2) = compute_mean_and_variance(&history, "mu2");

    println!("DSL Proposal - mean_mu1: {}", mean_mu1);
    println!("DSL Proposal - variance_mu1: {}", variance_mu1);

    println!("DSL Proposal - mean_mu2: {}", mean_mu2);
    println!("DSL Proposal - variance_mu2: {}", variance_mu2);

    let acceptance_rate = num_accepted as f64 / DRAW as f64;
    println!("DSL Proposal - acceptance_rate: {}", acceptance_rate);
    assert!(acceptance_rate > 0.1 && acceptance_rate < 0.9);

    assert!((mean_mu1 - mu1).abs() < 0.6);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.5);

    assert!((mean_mu2 - mu2).abs() < 0.6);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.5);
}
