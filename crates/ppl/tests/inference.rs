use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::{Bernoulli, Normal};
use std::sync::{Arc, Mutex};

use ppl::address::Address;
use ppl::dsl::trace::make_extract_args;
use ppl::dsl::Value;
use ppl::inference::metropolis_hastings_with_proposal;
use ppl::utils::compute_mean_and_variance;
use ppl::{r#gen, GenerativeFunction, Trace};
use ppl::dsl::trace::SchemeGenerativeFunction;
use ppl::dsl::trace::SchemeChoiceMap;

pub struct GMMParams {
    mu1: f64,
    mu2: f64,
    sigma1: f64,
    sigma2: f64,
    p: f64,
    num_samples: usize,
}

pub struct MHParams {
    burn_in: usize,
    num_samples: usize,
}



fn run_gmm_with_proposal(proposal: &SchemeGenerativeFunction, proposal_args: Vec<Value>, extract_args: Vec<Address>, gmm: GMMParams, mh_params: MHParams) {
    const SEED: u64 = 42;

    let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(SEED)));

    let z_dist = Bernoulli::new(gmm.p).unwrap();
    let z: Vec<bool> = (0..gmm.num_samples)
        .map(|_| z_dist.sample(&mut *rng.lock().unwrap()))
        .collect();

    let component1 = Normal::new(gmm.mu1, gmm.sigma1).unwrap();
    let c1: Vec<f64> = (0..gmm.num_samples)
        .map(|_| component1.sample(&mut *rng.lock().unwrap()))
        .collect();

    let component2 = Normal::new(gmm.mu2, gmm.sigma2).unwrap();
    let c2: Vec<f64> = (0..gmm.num_samples)
        .map(|_| component2.sample(&mut *rng.lock().unwrap()))
        .collect();

    let data = Value::List(
        (0..gmm.num_samples)
            .map(|i| if z[i] { c1[i] } else { c2[i] })
            .into_iter()
            .map(|x| Value::Float(x))
            .collect(),
    );

    let model = gen!([sigma1, sigma2, p, data] {
        // Priors
        (sample mu1 (normal 0.0 1.0))
        (sample mu2 (normal 0.0 1.0))

        // Ordering
        (constrain (< mu1 mu2))

        // Mixture
        (define mix (mixture (list (normal mu1 sigma1) (normal mu2 sigma2)) (list p (- 1.0 p))))

        (define observe-point (lambda (x) (observe (gensym) mix x)))

        (for-each observe-point data)

        (list mu1 mu2)
    });


    let args = vec![
        Value::Float(gmm.sigma1),
        Value::Float(gmm.sigma2),
        Value::Float(gmm.p),
        data.clone(),
    ];
    let extract_args = make_extract_args(extract_args);


    let mut trace = model.simulate(rng.clone(), args.clone());
    let mut attempts = 0;

    while !trace.get_score().is_finite() && attempts < 1000 {
        trace = model.simulate(rng.clone(), args.clone());
        attempts += 1;
    }

    if !trace.get_score().is_finite() {
        panic!(
            "Unable to initialise Markov chain: failed to obtain finite posterior score after {} attempts. Check whether the chosen parameters (e.g. very tight σ or extreme μ) are compatible with the prior.",
            attempts
        );
    }

    // Burn-in
    for _ in 0..mh_params.burn_in {
        let (new_trace, _) = metropolis_hastings_with_proposal(
            rng.clone(),
            trace,
            proposal,
            proposal_args.clone(),
            &extract_args,
            false,
            SchemeChoiceMap::new(),
        )
        .unwrap();

        trace = new_trace;
    }

    let mut history = Vec::with_capacity(mh_params.num_samples);
    let mut num_accepted = 0u32;

    // Sampling
    for _ in 0..mh_params.num_samples {
        let (new_trace, accepted) = metropolis_hastings_with_proposal(
            rng.clone(),
            trace,
            proposal,
            proposal_args.clone(),
            &extract_args,
            false,
            SchemeChoiceMap::new(),
        )
        .unwrap();

        history.push(new_trace.clone());

        trace = new_trace;

        if accepted {
            num_accepted += 1;
        }
    }

    let acceptance_rate = num_accepted as f64 / mh_params.num_samples as f64;
    let (mean_mu1, variance_mu1) = compute_mean_and_variance(&history, "mu1");
    let (mean_mu2, variance_mu2) = compute_mean_and_variance(&history, "mu2");

    // Check acceptance rate
    assert!(acceptance_rate > 0.1 && acceptance_rate < 0.9, "Acceptance rate {} is not in the range 0.1 to 0.9", acceptance_rate);

    // Check mean and variance
    assert!((mean_mu1 - gmm.mu1).abs() < 0.6, "Mean mu1 {} is not close to the true value {}", mean_mu1, gmm.mu1);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.5, "Variance mu1 {} is not in the range 0.0 to 2.5", variance_mu1);

    assert!((mean_mu2 - gmm.mu2).abs() < 0.6, "Mean mu2 {} is not close to the true value {}", mean_mu2, gmm.mu2);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.5, "Variance mu2 {} is not in the range 0.0 to 2.5", variance_mu2);

    
}



#[test]
fn test_gmm_with_symmetric_proposal() {
    // Parameters
    let mh_params = MHParams {
        burn_in: 100,
        num_samples: 1000,
    };

    let gmm = GMMParams {
        num_samples: 200,
        mu1: -2.0,
        mu2: 2.0,
        sigma1: 1.0,
        sigma2: 1.0,
        p: 0.5,
    };

    let extract_args = vec![
        Address::from("mu1"), 
        Address::from("mu2")
    ];

    // Define proposal
    let proposal_args = vec![
        Value::Float(0.15), // τ
    ];

    let proposal = gen!([tau, mu1, mu2] {
        (sample mu1 (normal mu1 tau))
        (sample mu2 (normal mu2 tau))
    });

    run_gmm_with_proposal(&proposal, proposal_args, extract_args, gmm, mh_params);
}


#[test]
fn test_gmm_with_asymmetric_proposal() {
    // Parameters
    let gmm = GMMParams {
        num_samples: 200,
        mu1: -5.0,
        mu2: 2.0,
        sigma1: 1.0,
        sigma2: 2.0,
        p: 0.3,
    };

    let mh_params = MHParams {
        burn_in: 500,
        num_samples: 2000,
    };

    let extract_args = vec![
        Address::from("mu1"), 
        Address::from("mu2")
    ];

    // Define proposal
    let proposal_args = vec![
        Value::Float(0.15), // τ_small
        Value::Float(3.0),  // τ_big
        Value::Float(0.10), // β
    ];

    let proposal = gen!([tau_s, tau_b, beta, mu1, mu2] {
        (sample big? (bernoulli beta))

        (define tau (if big? tau_b tau_s))

        (sample mu1 (normal mu1 tau))
        (sample mu2 (normal mu2 tau))
    });

    run_gmm_with_proposal(&proposal, proposal_args, extract_args, gmm, mh_params);
}