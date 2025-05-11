use std::collections::HashMap;
use std::collections::HashSet;

use rand::distributions::{Bernoulli, Distribution};
use statrs::distribution::Normal;

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
        name: "mu".to_string(),
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

    // Print initial mu
    if let Value::Float(mu) = trace.get_choice(&"mu".to_string()).value {
        println!("Initial mu: {}", mu);
    }

    let selection = HashSet::from_iter(vec!["mu".to_string()]);
    for i in 0..100 {
        let (new_trace, weight) = mh(program.clone(), trace, selection.clone()).unwrap();
        if i % 10 == 0 {
            if let Value::Float(mu) = new_trace.get_choice(&"mu".to_string()).value {
                println!("Warmup step {}: mu = {}, weight = {}", i, mu, weight);
            }
        }
        trace = new_trace;
    }

    let mut samples = Vec::new();
    for i in 0..100 {
        let (new_trace, accepted) = mh(program.clone(), trace, selection.clone()).unwrap();
        if i % 10 == 0 {
            if let Value::Float(mu) = new_trace.get_choice(&"mu".to_string()).value {
                println!("Sample step {}: mu = {}, accepted = {}", i, mu, accepted);
            }
        }
        if accepted {
            if let Value::Float(mu) = new_trace.get_choice(&"mu".to_string()).value {
                samples.push(mu);
            }
        }
        trace = new_trace;
    }

    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance: f64 =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

    println!("(Gaussian) mu mean: {:.3}, var: {:.3}", mean, variance);

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

    for (i, x) in data.iter().enumerate() {
        println!("(observe x{:?} mix {:?})", i, x)
    }

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
        name: "mu1".to_string(),
    });

    model.push(Expression::Sample {
        distribution: Box::new(Expression::List(vec![
            Expression::Variable("normal".to_string()),
            Expression::Constant(Literal::Float(0.0)),
            Expression::Constant(Literal::Float(1.0)),
        ])),
        name: "mu2".to_string(),
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

    // Print initial mu
    if let Value::Float(mu1) = trace.get_choice(&"mu1".to_string()).value {
        println!("Initial mu1: {}", mu1);
    }
    if let Value::Float(mu2) = trace.get_choice(&"mu2".to_string()).value {
        println!("Initial mu1: {}", mu2);
    }

    let selection = HashSet::from_iter(vec!["mu1".to_string(), "mu2".to_string()]);
    for i in 0..n {
        let (new_trace, weight) = mh(program.clone(), trace, selection.clone()).unwrap();
        if i % 10 == 0 {
            if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                println!("Warmup step {}: mu1 = {}, weight = {}", i, mu1, weight);
            }
            if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                println!("Warmup step {}: mu2 = {}, weight = {}", i, mu2, weight);
            }
        }
        trace = new_trace;
    }

    let mut samples_mu1 = Vec::new();
    let mut samples_mu2 = Vec::new();
    for i in 0..n {
        let (new_trace, accepted) = mh(program.clone(), trace, selection.clone()).unwrap();
        if i % 10 == 0 {
            if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                println!("Sample step {}: mu = {}, accepted = {}", i, mu1, accepted);
            }
            if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                println!("Sample step {}: mu = {}, accepted = {}", i, mu2, accepted);
            }
        }
        if accepted {
            if let Value::Float(mu1) = new_trace.get_choice(&"mu1".to_string()).value {
                samples_mu1.push(mu1);
            }
            if let Value::Float(mu2) = new_trace.get_choice(&"mu2".to_string()).value {
                samples_mu2.push(mu2);
            }
        }
        trace = new_trace;
    }

    println!("Samples Mean 1: {:?}", samples_mu1);

    let mean_mu1: f64 = samples_mu1.iter().sum::<f64>() / samples_mu1.len() as f64;

    let variance_mu1: f64 = samples_mu1
        .iter()
        .map(|x| (x - mean_mu1).powi(2))
        .sum::<f64>()
        / samples_mu1.len() as f64;

    println!(
        "(Gaussian) mu1 mean: {:.3}, var: {:.3}",
        mean_mu1, variance_mu1
    );

    assert!((mean_mu1 + 1.0).abs() < 0.5);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

    let mean_mu2: f64 = samples_mu2.iter().sum::<f64>() / samples_mu2.len() as f64;
    let variance_mu2: f64 = samples_mu2
        .iter()
        .map(|x| (x - mean_mu2).powi(2))
        .sum::<f64>()
        / samples_mu2.len() as f64;

    println!(
        "(Gaussian) mu2 mean: {:.3}, var: {:.3}",
        mean_mu2, variance_mu2
    );

    assert!((mean_mu2 - 1.0).abs() < 0.5);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
}

#[test]
fn test_gmm_parse() {
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
    let wrapped_data = Value::List(data.into_iter().map(|x| Value::Float(x)).collect());

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
    scales.insert(mu1_name.clone(), 1.0);
    scales.insert(mu2_name.clone(), 1.0);

    let program = GenerativeFunction::new(model, vec!["data".to_string()], scales, 42);

    let mut trace = program.simulate(vec![wrapped_data]).unwrap();

    // Print initial mu
    let mu1 = trace.get_choice(&mu1_name).value.expect_float();
    let mu2 = trace.get_choice(&mu2_name).value.expect_float();

    println!("Initial mu1: {}", mu1);
    println!("Initial mu2: {}", mu2);

    let selection = HashSet::from_iter(vec![mu1_name.clone(), mu2_name.clone()]);
    for _ in 0..n {
        let (new_trace, _) = mh(program.clone(), trace, selection.clone()).unwrap();
        trace = new_trace;
    }

    let mut history = Vec::new();
    let mut num_accepted = 0;
    for _ in 0..n {
        let (new_trace, accepted) = mh(program.clone(), trace, selection.clone()).unwrap();

        history.push(new_trace.clone());
        trace = new_trace;
        num_accepted += accepted as u32;
    }

    println!("Acceptence Rate: {:.3}", num_accepted as f64 / n as f64);

    let mean_mu1: f64 = history
        .iter()
        .map(|t| t.get_choice(&mu1_name).value.expect_float())
        .sum::<f64>()
        / history.len() as f64;
    let variance_mu1: f64 = history
        .iter()
        .map(|t| (t.get_choice(&mu1_name).value.expect_float() - mean_mu1).powi(2))
        .sum::<f64>()
        / history.len() as f64;

    println!(
        "(Gaussian) mu1 mean: {:.3}, var: {:.3}",
        mean_mu1, variance_mu1
    );

    let mean_mu2: f64 = history
        .iter()
        .map(|t| t.get_choice(&mu2_name).value.expect_float())
        .sum::<f64>()
        / history.len() as f64;
    let variance_mu2: f64 = history
        .iter()
        .map(|t| (t.get_choice(&mu2_name).value.expect_float() - mean_mu2).powi(2))
        .sum::<f64>()
        / history.len() as f64;

    println!(
        "(Gaussian) mu2 mean: {:.3}, var: {:.3}",
        mean_mu2, variance_mu2
    );

    assert!((mean_mu1 + 1.0).abs() < 0.5);
    assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

    assert!((mean_mu2 - 1.0).abs() < 0.5);
    assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
}
