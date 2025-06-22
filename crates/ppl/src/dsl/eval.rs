use rand::RngCore;
use std::cell::RefCell;
use std::rc::Rc;

use crate::address::Address;
use crate::dsl::ast::{Env, Expression, HostFn, Literal, Procedure, Value};
use crate::dsl::primitives::*;
use crate::dsl::trace::SchemeDSLTrace;
use crate::gfi::Trace;

/// Helper function to evaluate a distribution and either sample from it or score a value
/// This centralizes the common pattern of distribution evaluation used in Sample and Observe expressions
pub fn eval_distribution(
    distribution_value: Value,
    value: Option<&Value>, // None = sample, Some(value) = score
    rng: &mut dyn RngCore,
) -> Result<(Value, f64), String> {
    match distribution_value {
        Value::Procedure(Procedure::Stochastic {
            name: dist_name,
            args,
        }) => {
            let args = args.unwrap_or_default();
            let dist = create_distribution(&dist_name, &args)?;

            match value {
                None => {
                    // Sample mode
                    let sampled_value = dist.sample(rng);
                    let score = dist.log_prob(&sampled_value)?;
                    Ok((sampled_value, score))
                }
                Some(val) => {
                    // Score mode
                    let score = dist.log_prob(val)?;
                    Ok((val.clone(), score))
                }
            }
        }
        _ => Err("Distribution expression must yield a distribution".to_string()),
    }
}

/// Evaluates an expression against an environment, using the specified trace and random number generator.
/// The function is a monadic recursive evaluator, updating the trace with stochastic choices made during evaluation.
pub fn eval(
    expr: Expression,
    env: Rc<RefCell<Env>>,
    trace: &mut SchemeDSLTrace,
    rng: &mut dyn RngCore,
) -> Result<Value, String> {
    match expr {
        // Constants evaluate to their corresponding values
        Expression::Constant(c) => match c {
            Literal::Boolean(b) => Ok(Value::Boolean(b)),
            Literal::Integer(i) => Ok(Value::Integer(i)),
            Literal::Float(f) => Ok(Value::Float(f)),
            Literal::String(s) => Ok(Value::String(s)),
        },

        // Variables evaluate to their corresponding values in the environment
        Expression::Variable(name) => {
            let addr = Address::Symbol(name.clone());
            if trace.get_choices().contains(&addr) {
                if let Some(val) = trace.get_choice_value(&addr) {
                    return Ok(val);
                }
            }

            if let Some(value) = env.borrow().get(&name) {
                return Ok(value);
            }

            Err(format!("Unbound variable: {}", name))
        }

        Expression::List(exprs) => {
            // Evaluate each expression in the list
            let mut values = Vec::with_capacity(exprs.len());
            for e in &exprs {
                let val = eval(e.clone(), env.clone(), trace, rng)?;
                values.push(val);
            }

            apply(values[0].clone(), values[1..].to_vec(), trace, rng)
        }

        // Lambda creates closure
        Expression::Lambda(params, body) => Ok(Value::Procedure(Procedure::Lambda {
            params,
            body: body,
            env: env.clone(),
        })),

        // If evaluates the condition and executes the consequent or alternative
        Expression::If(cond, conseq, alt) => {
            let value = eval(*cond, env.clone(), trace, rng)?;
            match value {
                Value::Boolean(true) => eval(*conseq, env.clone(), trace, rng),
                Value::Boolean(false) => eval(*alt, env.clone(), trace, rng),
                _ => Err("Condition must be a boolean".to_string()),
            }
        }

        // Define binds a name to a value in the current environment
        Expression::Define(name, expr) => {
            // Extend env by binding the value of (eval(expr, env)) to name
            // Return the extended env
            let value = eval(*expr, env.clone(), trace, rng)?;
            env.borrow_mut().set(&name, value.clone());
            Ok(Value::Env(env))
        }

        // Quote returns the expression without evaluating it
        Expression::Quote(expr) => Ok(Value::Expr(*expr)),

        Expression::Sample { distribution, name } => {
            // Evaluate the name and distribution expressions
            let name = eval(*name, env.clone(), trace, rng)?;
            let dist = eval(*distribution, env.clone(), trace, rng)?;

            // Get the address of the name
            let addr = match name {
                Value::String(s) => s,
                Value::Procedure(_) | Value::List(_) | Value::Env(_) | Value::Expr(_) => {
                    return Err("sample: name must evaluate to a string".into())
                }
                other => format!("{:?}", other),
            };

            // Use the helper function to sample from the distribution
            let (value, score) = eval_distribution(dist, None, rng)?;

            // Add to trace
            let _ = trace.add_choice(Address::Symbol(addr), value.clone(), score);
            Ok(value)
        }

        Expression::Observe {
            name,
            distribution,
            observed,
        } => {
            // Evaluate both the distribution and observed value
            let name = eval(*name, env.clone(), trace, rng)?;
            let dist = eval(*distribution, env.clone(), trace, rng)?;
            let value = eval(*observed, env.clone(), trace, rng)?;

            let addr = match name {
                Value::String(s) => s,
                Value::Procedure(_) | Value::List(_) | Value::Env(_) | Value::Expr(_) => {
                    return Err("observe: name must evaluate to a string".into())
                }
                other => format!("{:?}", other),
            };

            // Use the helper function to score the observed value
            let (_, score) = eval_distribution(dist, Some(&value), rng)?;

            // Add to trace and return the observed value
            let _ = trace.add_choice(Address::Symbol(addr), value.clone(), score);
            Ok(value)
        }

        Expression::ForEach { func, seq } => {
            // Evaluate the procedure expression once
            let proc = eval(*func.clone(), env.clone(), trace, rng)?;

            // Evaluate the sequence expression once
            let seq = eval(*seq.clone(), env.clone(), trace, rng)?;

            let items = match seq {
                Value::List(v) => v,
                _ => return Err("for-each: second argument must be a list".into()),
            };

            // Call PROC on every element (spreading if the element itself is a list)
            for item in items {
                let arg = match item {
                    Value::List(v) => v,
                    v => vec![v],
                };
                apply(proc.clone(), arg, trace, rng)?;
            }

            // Return a Scheme-style "unit" – the empty list '()
            return Ok(Value::List(vec![]));
        }
    }
}

pub fn apply(
    func: Value,
    args: Vec<Value>,
    trace: &mut SchemeDSLTrace,
    rng: &mut dyn RngCore,
) -> Result<Value, String> {
    match func {
        Value::Procedure(Procedure::Deterministic { func }) => func(args),
        Value::Procedure(Procedure::Stochastic { name, args: _ }) => {
            Ok(Value::Procedure(Procedure::Stochastic {
                name,
                args: Some(args),
            }))
        }
        Value::Procedure(Procedure::Lambda {
            params,
            body,
            env: closure_env,
        }) => {
            if params.len() != args.len() {
                return Err(format!(
                    "Expected {} arguments, got {}",
                    params.len(),
                    args.len()
                ));
            }

            let new_env = Rc::new(RefCell::new(Env::with_parent(closure_env)));

            for (param, arg) in params.iter().zip(args) {
                new_env.borrow_mut().set(param, arg);
            }

            eval(*body, new_env, trace, rng)
        }

        other => Err(format!("{:?} is not a function", other)),
    }
}

fn wrap(f: fn(Vec<Value>) -> Result<Value, String>) -> HostFn {
    Rc::new(move |args| f(args))
}

pub fn standard_env() -> Rc<RefCell<Env>> {
    let env = Rc::new(RefCell::new(Env::new()));

    // Deterministic Primitives
    env.borrow_mut().set(
        "make-gensym",
        Value::Procedure(Procedure::Deterministic {
            func: wrap(make_gensym),
        }),
    );

    env.borrow_mut().set(
        "display",
        Value::Procedure(Procedure::Deterministic {
            func: wrap(display),
        }),
    );

    // Arithmetic Ops
    env.borrow_mut().set(
        "+",
        Value::Procedure(Procedure::Deterministic { func: wrap(add) }),
    );
    env.borrow_mut().set(
        "-",
        Value::Procedure(Procedure::Deterministic { func: wrap(sub) }),
    );
    env.borrow_mut().set(
        "*",
        Value::Procedure(Procedure::Deterministic { func: wrap(mul) }),
    );
    env.borrow_mut().set(
        "/",
        Value::Procedure(Procedure::Deterministic { func: wrap(div) }),
    );

    // Logical Ops
    env.borrow_mut().set(
        "=",
        Value::Procedure(Procedure::Deterministic { func: wrap(eq) }),
    );

    env.borrow_mut().set(
        "<",
        Value::Procedure(Procedure::Deterministic { func: wrap(lt) }),
    );

    // Unary Ops
    env.borrow_mut().set(
        "exp",
        Value::Procedure(Procedure::Deterministic { func: wrap(exp) }),
    );

    // Data Structures
    env.borrow_mut().set(
        "list",
        Value::Procedure(Procedure::Deterministic { func: wrap(list) }),
    );

    env.borrow_mut().set(
        "car",
        Value::Procedure(Procedure::Deterministic { func: wrap(car) }),
    );

    env.borrow_mut().set(
        "cdr",
        Value::Procedure(Procedure::Deterministic { func: wrap(cdr) }),
    );

    // Distribution Primitives
    env.borrow_mut().set(
        "normal",
        Value::Procedure(Procedure::Stochastic {
            name: "normal".to_string(),
            args: None,
        }),
    );

    env.borrow_mut().set(
        "exponential",
        Value::Procedure(Procedure::Stochastic {
            name: "exponential".to_string(),
            args: None,
        }),
    );

    env.borrow_mut().set(
        "mixture",
        Value::Procedure(Procedure::Stochastic {
            name: "mixture".to_string(),
            args: None,
        }),
    );

    env.borrow_mut().set(
        "condition",
        Value::Procedure(Procedure::Stochastic {
            name: "condition".to_string(),
            args: None,
        }),
    );

    env
}
