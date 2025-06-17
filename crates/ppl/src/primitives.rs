use crate::ast::{Procedure, Value};
use rand::RngCore;

use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::ast::HostFn;
use crate::distributions::{Condition, DistributionExtended, Mixture};

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

pub fn exp(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("exp expects 1 argument".to_string());
    }

    let v = args[0].as_float().ok_or("exp expects a numeric argument")?;
    Ok(Value::Float(v.exp()))
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

pub fn car(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("car expects exactly 1 argument".to_string());
    }

    match &args[0] {
        Value::List(list) => {
            if list.is_empty() {
                Err("car: cannot take car of empty list".to_string())
            } else {
                Ok(list[0].clone())
            }
        }
        _ => Err("car expects a list argument".to_string()),
    }
}

pub fn cdr(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cdr expects exactly 1 argument".to_string());
    }

    match &args[0] {
        Value::List(list) => {
            if list.is_empty() {
                Err("cdr: cannot take cdr of empty list".to_string())
            } else {
                Ok(Value::List(list[1..].to_vec()))
            }
        }
        _ => Err("cdr expects a list argument".to_string()),
    }
}

pub trait Parseable: Sized {
    fn parse(args: &[Value]) -> Result<Self, String>;
}

impl Parseable for statrs::distribution::Bernoulli {
    fn parse(args: &[Value]) -> Result<Self, String> {
        if args.len() != 1 {
            return Err(format!("Bernoulli expects 1 argument, got {}", args.len()));
        }

        let p = args[0]
            .as_float()
            .ok_or_else(|| "Bernoulli parameter must be a number".to_string())?;

        if p < 0.0 || p > 1.0 {
            return Err(format!(
                "Bernoulli parameter must be between 0 and 1, got {}",
                p
            ));
        }

        statrs::distribution::Bernoulli::new(p)
            .map_err(|e| format!("Failed to create Bernoulli distribution: {}", e))
    }
}

impl Parseable for Condition {
    fn parse(args: &[Value]) -> Result<Self, String> {
        if args.len() != 1 {
            return Err(format!("Condition expects 1 argument, got {}", args.len()));
        }

        let flag = args[0]
            .as_bool()
            .ok_or_else(|| "Condition argument must be a boolean".to_string())?;

        Ok(Condition::new(flag))
    }
}

impl Parseable for statrs::distribution::Normal {
    fn parse(args: &[Value]) -> Result<Self, String> {
        if args.len() != 2 {
            return Err(format!("Normal expects 2 arguments, got {}", args.len()));
        }

        let mean = args[0]
            .as_float()
            .ok_or_else(|| "Normal mean must be a number".to_string())?;
        let std_dev = args[1]
            .as_float()
            .ok_or_else(|| "Normal standard deviation must be a number".to_string())?;

        if std_dev <= 0.0 {
            return Err(format!(
                "Normal standard deviation must be positive, got {}",
                std_dev
            ));
        }

        statrs::distribution::Normal::new(mean, std_dev)
            .map_err(|e| format!("Failed to create Normal distribution: {}", e))
    }
}

impl Parseable for statrs::distribution::Exp {
    fn parse(args: &[Value]) -> Result<Self, String> {
        if args.len() != 1 {
            return Err(format!(
                "Exponential expects 1 argument, got {}",
                args.len()
            ));
        }

        let lambda = args[0]
            .as_float()
            .ok_or_else(|| "Exponential parameter must be a number".to_string())?;

        if lambda <= 0.0 {
            return Err(format!(
                "Exponential parameter must be positive, got {}",
                lambda
            ));
        }

        statrs::distribution::Exp::new(lambda)
            .map_err(|e| format!("Failed to create Exponential distribution: {}", e))
    }
}

impl Parseable for Mixture<f64> {
    fn parse(args: &[Value]) -> Result<Self, String> {
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
            let wv = w
                .as_float()
                .ok_or_else(|| "Mixture weights must be a number".to_string())?;
            if wv <= 0.0 {
                return Err("mixture: weights must be positive".into());
            }
            log_weights.push(wv.ln());
        }

        // Build each component by parsing its normal parameters
        let mut components: Vec<Box<dyn DistributionExtended<f64>>> =
            Vec::with_capacity(dist_specs.len());

        for spec in dist_specs {
            let dist = match spec {
                Value::Procedure(Procedure::Stochastic { name, args, .. }) => {
                    let args = args.clone().unwrap_or_default();
                    let dist = create_distribution(&name, &args)?;
                    dist.as_continuous().unwrap().clone_box()
                }
                _ => return Err("mixture: each component must be a list of parameters".into()),
            };

            components.push(dist);
        }
        // Return a Mixture distribution over f64
        Ok(Mixture {
            log_weights,
            components,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Enum representing different types of distributions with their appropriate output types
pub enum ValueDistribution {
    Continuous(Box<dyn DistributionExtended<f64>>),
    Discrete(Box<dyn DistributionExtended<bool>>),
}

impl ValueDistribution {
    /// Get the distribution as a continuous (f64) distribution if possible
    pub fn as_continuous(&self) -> Result<&Box<dyn DistributionExtended<f64>>, String> {
        match self {
            ValueDistribution::Continuous(dist) => Ok(dist),
            _ => Err("Not a continuous distribution".to_string()),
        }
    }

    /// Get the distribution as a discrete (bool) distribution if possible
    pub fn as_discrete(&self) -> Result<&Box<dyn DistributionExtended<bool>>, String> {
        match self {
            ValueDistribution::Discrete(dist) => Ok(dist),
            _ => Err("Not a discrete distribution".to_string()),
        }
    }

    /// Sample from the distribution, returning a Value
    pub fn sample(&self, rng: &mut dyn RngCore) -> Value {
        match self {
            ValueDistribution::Continuous(dist) => Value::Float(dist.sample_dyn(rng)),
            ValueDistribution::Discrete(dist) => Value::Boolean(dist.sample_dyn(rng)),
        }
    }

    /// Compute the log probability of a value
    pub fn log_prob(&self, value: &Value) -> Result<f64, String> {
        match (self, value) {
            (ValueDistribution::Continuous(dist), Value::Float(v)) => Ok(dist.log_prob(*v)),
            (ValueDistribution::Discrete(dist), Value::Boolean(v)) => Ok(dist.log_prob(*v)),
            _ => Err(format!(
                "Type mismatch: {:?} is not compatible with this distribution",
                value
            )),
        }
    }

    /// Returns true if the distribution is continuous (produces f64 values)
    pub fn is_continuous(&self) -> bool {
        matches!(self, ValueDistribution::Continuous(_))
    }

    /// Returns true if the distribution is discrete (produces bool values)
    pub fn is_discrete(&self) -> bool {
        matches!(self, ValueDistribution::Discrete(_))
    }
}

/// Creates a distribution of the appropriate type based on the name and arguments
pub fn create_distribution(name: &str, args: &[Value]) -> Result<ValueDistribution, String> {
    match name.to_lowercase().as_str() {
        // Continuous (f64) distributions
        "normal" => {
            let normal = statrs::distribution::Normal::parse(args)?;
            Ok(ValueDistribution::Continuous(Box::new(normal)))
        }
        "exponential" => {
            let exponential = statrs::distribution::Exp::parse(args)?;
            Ok(ValueDistribution::Continuous(Box::new(exponential)))
        }
        "mixture" => {
            let mixture = Mixture::parse(args)?;
            Ok(ValueDistribution::Continuous(Box::new(mixture)))
        }

        // Discrete (bool) distributions
        "bernoulli" => {
            let bernoulli = statrs::distribution::Bernoulli::parse(args)?;
            Ok(ValueDistribution::Discrete(Box::new(bernoulli)))
        }
        "condition" => {
            let condition = Condition::parse(args)?;
            Ok(ValueDistribution::Discrete(Box::new(condition)))
        }

        // Unknown distribution
        _ => Err(format!("Unknown distribution: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::statistics::Distribution;

    #[test]
    fn test_parse() {
        // Test Bernoulli parsing
        let bernoulli_args = vec![Value::Float(0.7)];
        let bernoulli = statrs::distribution::Bernoulli::parse(&bernoulli_args).unwrap();
        assert_eq!(bernoulli.p(), 0.7);

        // Test Normal parsing
        let normal_args = vec![Value::Float(2.0), Value::Float(1.5)];
        let normal = statrs::distribution::Normal::parse(&normal_args).unwrap();
        assert_eq!(normal.mean().unwrap(), 2.0);
        assert_eq!(normal.std_dev().unwrap(), 1.5);

        // Test Condition parsing
        let condition_args = vec![Value::Boolean(true)];
        let condition = Condition::parse(&condition_args).unwrap();
        assert_eq!(condition.flag, true);
    }

    #[test]
    fn test_create_distribution() {
        // Test creating a normal distribution (continuous/f64)
        let normal_args = vec![Value::Float(0.0), Value::Float(1.0)];
        let normal_dist = create_distribution("normal", &normal_args).unwrap();
        assert!(normal_dist.as_continuous().is_ok());
        assert!(normal_dist.as_discrete().is_err());
        assert!(normal_dist.is_continuous());
        assert!(!normal_dist.is_discrete());

        // Test creating with case-insensitive name
        let normal_dist2 = create_distribution("NoRmAl", &normal_args).unwrap();
        assert!(normal_dist2.as_continuous().is_ok());

        // Test creating a Bernoulli distribution (discrete/bool)
        let bernoulli_args = vec![Value::Float(0.7)];
        let bernoulli_dist = create_distribution("bernoulli", &bernoulli_args).unwrap();
        assert!(bernoulli_dist.as_discrete().is_ok());
        assert!(bernoulli_dist.as_continuous().is_err());
        assert!(!bernoulli_dist.is_continuous());
        assert!(bernoulli_dist.is_discrete());

        // Test creating a Condition distribution (discrete/bool)
        let condition_args = vec![Value::Boolean(true)];
        let condition_dist = create_distribution("condition", &condition_args).unwrap();
        assert!(condition_dist.as_discrete().is_ok());
        assert!(condition_dist.as_continuous().is_err());

        // Test error for unknown distribution
        let result = create_distribution("unknown_dist", &normal_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_distribution_methods() {
        // Test normal distribution sampling and log_prob
        let normal_args = vec![Value::Float(0.0), Value::Float(1.0)];
        let normal_dist = create_distribution("normal", &normal_args).unwrap();

        let mut rng = rand::thread_rng();
        let sample = normal_dist.sample(&mut rng);
        assert!(matches!(sample, Value::Float(_)));

        if let Value::Float(_) = sample {
            let log_prob = normal_dist.log_prob(&sample);
            assert!(log_prob.is_ok());
            assert!(log_prob.unwrap().is_finite());

            // Test type mismatch error
            let log_prob_err = normal_dist.log_prob(&Value::Boolean(true));
            assert!(log_prob_err.is_err());
        }

        // Test Bernoulli distribution sampling and log_prob
        let bernoulli_args = vec![Value::Float(0.7)];
        let bernoulli_dist = create_distribution("bernoulli", &bernoulli_args).unwrap();

        let sample = bernoulli_dist.sample(&mut rng);
        assert!(matches!(sample, Value::Boolean(_)));

        if let Value::Boolean(_) = sample {
            let log_prob = bernoulli_dist.log_prob(&sample);
            assert!(log_prob.is_ok());
            assert!(log_prob.unwrap().is_finite());

            // Test type mismatch error
            let log_prob_err = bernoulli_dist.log_prob(&Value::Float(0.5));
            assert!(log_prob_err.is_err());
        }
    }

    #[test]
    fn test_create_mixture_distribution() {
        // Create a Normal distribution as a component
        let normal_args = vec![Value::Float(0.0), Value::Float(1.0)];
        let normal_proc = Value::Procedure(Procedure::Stochastic {
            args: Some(normal_args),
            name: "normal".to_string(),
        });

        // Create a second Normal distribution as a component
        let normal2_args = vec![Value::Float(5.0), Value::Float(2.0)];
        let normal2_proc = Value::Procedure(Procedure::Stochastic {
            args: Some(normal2_args),
            name: "normal".to_string(),
        });

        // Create the mixture with equal weights
        let components = vec![normal_proc, normal2_proc];
        let weights = vec![Value::Float(0.5), Value::Float(0.5)];
        let mixture_args = vec![Value::List(components), Value::List(weights)];

        let mixture_dist = create_distribution("mixture", &mixture_args).unwrap();
        assert!(mixture_dist.is_continuous());
    }
}
