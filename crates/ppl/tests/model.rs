use std::collections::HashMap;
use std::collections::HashSet;

use rand::distributions::{Bernoulli, Distribution};
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::Normal;

use ppl::utils::compute_mean_and_variance;
use ppl::{mh, r#gen, Expression, GenerativeFunction, Literal, Value};

#[test]
fn test_univariate_gaussian_no_parse() {
    let mut rng = rand::thread_rng();
    let data_dist = Normal::new(5.0, 1.0).unwrap();
    let data: Vec<f64> = (0..100).map(|_| data_dist.sample(&mut rng)).collect();

    let mut model = vec![Expression::Sample {
        distribution: Box::new(Expression::List(vec![
            Expression::Variable("normal".to_string()),
            Expression::Constant(Literal::Float(0.0)),
            Expression::Constant(Literal::Float(1.0)),
        ])),
        name: Box::new(Expression::Constant(Literal::String("mu".to_string()))),
    }];

    for (i, &x) in data.iter().enumerate() {
        model.push(Expression::Observe {
            distribution: Box::new(Expression::List(vec![
                Expression::Variable("normal".to_string()),
                Expression::Variable("mu".to_string()),
                Expression::Constant(Literal::Float(1.0)),
            ])),
            observed: Box::new(Expression::Constant(Literal::Float(x))),
            name: Box::new(Expression::Constant(Literal::String(format!("x{}", i)))),
        });
    }

    let mut scales = HashMap::new();
    scales.insert("mu".to_string(), 1.0);

    let program = GenerativeFunction::new(model, vec![], scales, 42);

    let mut trace = program.simulate(vec![]).unwrap();

    let selection = HashSet::from_iter(vec!["mu".to_string()]);
    for _ in 0..100 {
        let (t, _) = mh(&program, trace, &selection).unwrap();
        trace = t;
    }

    let mut history = Vec::new();
    for _ in 0..100 {
        let (t, _) = mh(&program, trace, &selection).unwrap();
        trace = t;
        history.push(trace.clone());
    }

    let (mean, variance) = compute_mean_and_variance(&history, &"mu".to_string());

    assert!((mean - 5.0).abs() < 0.5);
    assert!(variance > 0.0 && variance < 2.0);
}

#[test]
fn test_gmm_no_parse() {
    let mut rng = rand::thread_rng();

    let n = 1000;
    let num_samples = 200;
    let mu1 = 1.0;
    let mu2 = -1.0;
    let sigma = 1.0;
    let p = 0.5;
    let z_dist = Bernoulli::new(p).unwrap();
    let z: Vec<bool> = (0..num_samples).map(|_| z_dist.sample(&mut rng)).collect();

    let component1 = Normal::new(mu1, sigma).unwrap();
    let c1: Vec<f64> = (0..num_samples)
        .map(|_| component1.sample(&mut rng))
        .collect();

    let component2 = Normal::new(mu2, sigma).unwrap();
    let c2: Vec<f64> = (0..num_samples)
        .map(|_| component2.sample(&mut rng))
        .collect();

    let data: Vec<f64> = (0..num_samples)
        .map(|i| if z[i] { c1[i] } else { c2[i] })
        .collect();

    // Define model
    let mut model = vec![Expression::Define(
        "p".to_string(),
        Box::new(Expression::Constant(Literal::Float(p))),
    )];

    model.push(Expression::Sample {
        distribution: Box::new(Expression::List(vec![
            Expression::Variable("normal".to_string()),
            Expression::Constant(Literal::Float(0.0)),
            Expression::Constant(Literal::Float(1.0)),
        ])),
        name: Box::new(Expression::Constant(Literal::String("mu1".to_string()))),
    });

    model.push(Expression::Sample {
        distribution: Box::new(Expression::List(vec![
            Expression::Variable("normal".to_string()),
            Expression::Constant(Literal::Float(0.0)),
            Expression::Constant(Literal::Float(1.0)),
        ])),
        name: Box::new(Expression::Constant(Literal::String("mu2".to_string()))),
    });

    let logic = Expression::List(vec![
        Expression::Variable("<".to_string()),
        Expression::Variable("mu1".into()),
        Expression::Variable("mu2".into()),
    ]);

    model.push(Expression::Observe {
        distribution: Box::new(Expression::List(vec![
            Expression::Variable("condition".into()),
            logic,
        ])),
        observed: Box::new(Expression::Constant(Literal::Boolean(true))),
        name: Box::new(Expression::Constant(Literal::String(
            "mu1-lt-mu2".to_string(),
        ))),
    });

    let mixture = Expression::List(vec![
        Expression::Variable("mixture".into()),
        Expression::List(vec![
            Expression::Variable("list".into()),
            Expression::List(vec![
                Expression::Variable("normal".into()),
                Expression::Variable("mu1".into()),
                Expression::Constant(Literal::Float(1.0)),
            ]),
            Expression::List(vec![
                Expression::Variable("normal".into()),
                Expression::Variable("mu2".into()),
                Expression::Constant(Literal::Float(1.0)),
            ]),
        ]),
        Expression::List(vec![
            Expression::Variable("list".into()),
            Expression::Variable("p".into()),
            Expression::List(vec![
                Expression::Variable("-".into()),
                Expression::Constant(Literal::Float(1.0)),
                Expression::Variable("p".into()),
            ]),
        ]),
    ]);

    model.push(Expression::Define("mix".into(), Box::new(mixture)));

    for (i, &x) in data.iter().enumerate() {
        model.push(Expression::Observe {
            distribution: Box::new(Expression::Variable("mix".into())), //Box::new(mixture.clone()),
            observed: Box::new(Expression::Constant(Literal::Float(x))),
            name: Box::new(Expression::Constant(Literal::String(format!("x{}", i)))),
        });
    }

    let mut scales = HashMap::new();
    scales.insert("mu1".to_string(), 1.0);
    scales.insert("mu2".to_string(), 1.0);

    let program = GenerativeFunction::new(model, vec![], scales, 42);

    let mut trace = program.simulate(vec![]).unwrap();

    let selection = HashSet::from_iter(vec!["mu1".to_string(), "mu2".to_string()]);
    for _ in 0..n {
        let (t, _) = mh(&program, trace, &selection).unwrap();
        trace = t;
    }

    let mut history = Vec::with_capacity(n);
    for _ in 0..n {
        let (t, _) = mh(&program, trace, &selection).unwrap();
        trace = t;
        history.push(trace.clone());
    }

    let (mean_mu1, variance_mu1) = compute_mean_and_variance(&history, &"mu1".to_string());
    let (mean_mu2, variance_mu2) = compute_mean_and_variance(&history, &"mu2".to_string());

    assert!((mean_mu1 + 1.0).abs() < 0.5);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

    assert!((mean_mu2 - 1.0).abs() < 0.5);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
}

#[test]
fn test_gmm_parse() {
    const SEED: u64 = 42;
    const BURN_IN: usize = 100;
    const DRAW: usize = 1000;
    const PROP_SD: f64 = 0.15;

    let data_seed = 40;
    let mut rng = StdRng::seed_from_u64(data_seed);

    let num_samples = 500;
    let mu1 = -2.0;
    let mu2 = 2.0;
    let sigma = 1.0;
    let p = 0.5;
    let z_dist = Bernoulli::new(p).unwrap();
    let z: Vec<bool> = (0..num_samples).map(|_| z_dist.sample(&mut rng)).collect();

    let component1 = Normal::new(mu1, sigma).unwrap();
    let c1: Vec<f64> = (0..num_samples)
        .map(|_| component1.sample(&mut rng))
        .collect();

    let component2 = Normal::new(mu2, sigma).unwrap();
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

    let model = gen!(
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
    );

    let mu1_name = "mu1".to_string();
    let mu2_name = "mu2".to_string();

    let mut scales = HashMap::new();
    scales.insert(mu1_name.clone(), PROP_SD);
    scales.insert(mu2_name.clone(), PROP_SD);

    let program = GenerativeFunction::new(model, vec!["data".to_string()], scales, SEED);

    let selection = HashSet::from_iter(vec![mu1_name.clone(), mu2_name.clone()]);
    let mut trace = program.simulate(vec![data]).unwrap();
    for _ in 0..BURN_IN {
        let (t, _) = mh(&program, trace, &selection).unwrap();
        trace = t;
    }

    let mut history = Vec::with_capacity(DRAW);
    let mut num_accepted = 0u32;

    for _ in 0..DRAW {
        let (t, accepted) = mh(&program, trace, &selection).unwrap();
        trace = t;
        if accepted {
            num_accepted += 1;
        }
        history.push(trace.clone());
    }

    let acceptance_rate = num_accepted as f64 / DRAW as f64;
    assert!(acceptance_rate > 0.2 && acceptance_rate < 0.8);

    let (mean_mu1, variance_mu1) = compute_mean_and_variance(&history, &mu1_name);
    let (mean_mu2, variance_mu2) = compute_mean_and_variance(&history, &mu2_name);

    assert!((mean_mu1 - mu1).abs() < 0.5);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

    assert!((mean_mu2 - mu2).abs() < 0.5);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
}
