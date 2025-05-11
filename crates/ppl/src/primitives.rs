use rand::RngCore;

use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::ast::{HostFn, Procedure, Value};
use crate::distributions::{Condition, DistributionExtended, Mixture};

// TODO: Fix the distribution primitives to be more unified.

/// Make a gensym.
pub fn make_gensym(args: Vec<Value>) -> Result<Value, String> {
    let prefix = match args.as_slice() {
        [] => "g".to_string(),
        [Value::String(s)] => s.clone(),
        _ => return Err("make-gensym: expects zero or one string argument".into()),
    };

    let counter = Arc::new(AtomicUsize::new(0));

    let closure: HostFn = {
        let ctr = counter.clone();
        let pref = prefix.clone();
        Rc::new(move |_noargs| {
            if !_noargs.is_empty() {
                return Err("gensym takes no arguments".into());
            }
            let id = ctr.fetch_add(1, Ordering::Relaxed);

            Ok(Value::String(format!("{}{}", pref, id)))
        })
    };

    Ok(Value::Procedure(Procedure::Deterministic { func: closure }))
}

// Helper to convert Value to f64 and track if it was originally a float
fn get_numeric(val: &Value) -> Result<(f64, bool), String> {
    match val {
        Value::Integer(n) => Ok((*n as f64, false)),
        Value::Float(f) => Ok((*f, true)),
        _ => Err("Expected numeric (integer or float) arguments".to_string()),
    }
}

// Helper to create the final Value based on the result and whether floats were involved
fn finalize_numeric_result(result: f64, saw_float: bool) -> Value {
    // If we saw a float OR the result has a fractional part, return Float
    if saw_float || result.fract() != 0.0 {
        Value::Float(result)
    } else {
        Value::Integer(result as i64)
    }
}

pub fn add(args: Vec<Value>) -> Result<Value, String> {
    args.iter()
        .try_fold((0.0, false), |(acc, saw_float_acc), arg| {
            let (val, is_float_arg) = get_numeric(arg)?;
            Ok((acc + val, saw_float_acc || is_float_arg))
        })
        .map(|(sum, saw_float)| finalize_numeric_result(sum, saw_float))
}

pub fn sub(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("- requires at least one argument".to_string());
    }

    let (mut result, mut saw_float) = get_numeric(&args[0])?;

    // Handle unary minus case: (- 5) -> -5
    if args.len() == 1 {
        result = -result;
    } else {
        // Handle multi-argument case: (- 10 2 3) -> 10 - 2 - 3 = 5
        for arg in &args[1..] {
            let (val, is_float) = get_numeric(arg)?;
            result -= val;
            saw_float = saw_float || is_float;
        }
    }

    Ok(finalize_numeric_result(result, saw_float))
}

pub fn mul(args: Vec<Value>) -> Result<Value, String> {
    args.iter()
        .try_fold((1.0, false), |(acc, saw_float_acc), arg| {
            let (val, is_float_arg) = get_numeric(arg)?;
            Ok((acc * val, saw_float_acc || is_float_arg))
        })
        .map(|(product, saw_float)| finalize_numeric_result(product, saw_float))
}

pub fn div(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("/ requires at least one argument".to_string());
    }

    let (mut result, mut saw_float) = get_numeric(&args[0])?;

    // Handle unary division case: (/ 5) -> 1/5
    if args.len() == 1 {
        if result == 0.0 {
            return Err("Division by zero".to_string());
        }
        result = 1.0 / result;
        // Check if the original was float OR 1/int produces float
        saw_float = saw_float || result.fract() != 0.0;
    } else {
        // Handle multi-argument case: (/ 10 2 5) -> 10 / 2 / 5 = 1
        for arg in &args[1..] {
            let (val, is_float) = get_numeric(arg)?;
            if val == 0.0 {
                return Err("Division by zero".to_string());
            }
            result /= val;
            saw_float = saw_float || is_float;
        }
    }

    Ok(finalize_numeric_result(result, saw_float))
}

pub fn eq(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("= takes exactly two arguments".to_string());
    }

    let (a, b) = (&args[0], &args[1]);

    match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(*a == *b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(*a == *b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Boolean(*a == *b)),
        (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a == *b)),
        _ => Err("= expects numeric or string arguments".to_string()),
    }
}

pub fn lt(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("< takes exactly two arguments".to_string());
    }

    let (a, b) = (&args[0], &args[1]);

    match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(*a < *b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(*a < *b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Boolean(*a < *b)),
        (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a < *b)),
        _ => Err("= expects numeric or string arguments".to_string()),
    }
}

pub fn list(args: Vec<Value>) -> Result<Value, String> {
    return Ok(Value::List(args));
}

pub fn display(v: Vec<Value>) -> Result<Value, String> {
    print!("{:?}", v);
    Ok(Value::List(vec![]))
}

// Distribution primitives
fn parse_normal_args(args: &[Value]) -> Result<Box<dyn DistributionExtended<f64>>, String> {
    use statrs::distribution::Normal;

    match args.len() {
        0 => Ok(Box::new(Normal::new(0.0, 1.0).unwrap())),
        1 => match &args[0] {
            Value::Float(f) => Ok(Box::new(Normal::new(*f, 1.0).unwrap())),
            Value::Integer(i) => Ok(Box::new(Normal::new(*i as f64, 1.0).unwrap())),
            _ => Err("normal: expected numeric mean".into()),
        },
        2 => {
            let (mean, std) = (&args[0], &args[1]);
            match (mean, std) {
                (Value::Float(mean), Value::Float(std)) => {
                    Ok(Box::new(Normal::new(*mean, *std).unwrap()))
                }
                (Value::Integer(mean), Value::Integer(std)) => {
                    Ok(Box::new(Normal::new(*mean as f64, *std as f64).unwrap()))
                }
                _ => Err("normal: expected numeric mean and std".into()),
            }
        }
        _ => Err("normal: expected 0 or 2 arguments".into()),
    }
}

pub fn normal_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_normal_args(&args)?;
    Ok(Value::Float(dist.sample_dyn(rng)))
}

pub fn normal_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
    let dist = parse_normal_args(&args)?;
    match value {
        Value::Float(f) => Ok(dist.log_prob(f)),
        Value::Integer(i) => Ok(dist.log_prob(i as f64)),
        _ => Err("normal log_prob expects a numeric value".into()),
    }
}

fn parse_mixture_args(args: &[Value]) -> Result<Box<dyn DistributionExtended<f64>>, String> {
    // Args: Vec<Stochastic Procedure> Vec<probabilities>
    // Expect exactly two arguments: a list of component specs and a list of weights
    if args.len() != 2 {
        return Err(
            "mixture: expected 2 arguments: list of distributions and list of weights".into(),
        );
    }
    // First argument: list of component specifications (each itself a list of normal parameters)
    let dist_specs = match &args[0] {
        Value::List(v) => v,
        _ => {
            return Err(
                "mixture: first argument must be a list of distribution parameter lists".into(),
            )
        }
    };
    // Second argument: list of numeric weights
    let weight_vals = match &args[1] {
        Value::List(v) => v,
        _ => return Err("mixture: second argument must be a list of numeric weights".into()),
    };
    if dist_specs.len() != weight_vals.len() {
        return Err("mixture: number of distributions and weights must match".into());
    }
    // Convert weights to log-space
    let mut log_weights = Vec::with_capacity(weight_vals.len());
    for w in weight_vals {
        let (wv, _) = get_numeric(w)?;
        if wv <= 0.0 {
            return Err("mixture: weights must be positive".into());
        }
        log_weights.push(wv.ln());
    }

    // Build each component by parsing its normal parameters
    let mut components: Vec<Box<dyn DistributionExtended<f64>>> =
        Vec::with_capacity(dist_specs.len());
    for spec in dist_specs {
        let params = match spec {
            Value::Procedure(Procedure::Stochastic { args, .. }) => args.clone().unwrap(),
            _ => return Err("mixture: each component must be a list of parameters".into()),
        };

        // TODO: Should work for generic distributions
        let dist = parse_normal_args(&params)?;
        components.push(dist);
    }
    // Return a Mixture distribution over f64
    Ok(Box::new(Mixture {
        log_weights,
        components,
        _marker: std::marker::PhantomData,
    }))
}

pub fn mixture_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_mixture_args(&args)?;
    Ok(Value::Float(dist.sample_dyn(rng)))
}

pub fn mixture_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
    let dist = parse_mixture_args(&args)?;
    match value {
        Value::Float(f) => Ok(dist.log_prob(f)),
        Value::Integer(i) => Ok(dist.log_prob(i as f64)),
        _ => Err("mixture log_prob expects a numeric value".into()),
    }
}

fn parse_condition_args(args: &[Value]) -> Result<Box<dyn DistributionExtended<bool>>, String> {
    match args.len() {
        0 => Ok(Box::new(Condition::new(true))),
        1 => {
            let value = &args[0];

            match value {
                Value::Boolean(flag) => Ok(Box::new(Condition::new(*flag))),
                _ => Err("condition expects a boolean paramter".into()),
            }
        }
        _ => Err("condition only expects 1 parameter".into()),
    }
}

pub fn condition_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_condition_args(&args)?;
    Ok(Value::Boolean(dist.sample_dyn(rng)))
}

pub fn condition_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
    let dist = parse_condition_args(&args)?;

    match value {
        Value::Boolean(x) => Ok(dist.log_prob(x)),
        _ => Err("Invalid input for domain boolean".into()),
    }
}
