use std::cell::RefCell;
use std::rc::Rc;

use rand::RngCore;

use crate::address::Address;
use crate::dsl::ast::{Env, Expression, HostFn, Literal, Procedure, Value};
use crate::dsl::primitives::*;

/// Represents a choice event that occurs during evaluation
#[derive(Debug, Clone)]
pub struct ChoiceEvent {
    pub address: Address,
    pub dist_name: String,
    pub dist_args: Vec<Value>,
    pub obs: Option<Value>,
}

impl ChoiceEvent {
    pub fn new(
        address: Address,
        dist_name: String,
        dist_args: Vec<Value>,
        obs: Option<Value>,
    ) -> Self {
        Self {
            address,
            dist_name,
            dist_args,
            obs,
        }
    }

    pub fn sample(&self, rng: &mut dyn RngCore, dist: &ValueDistribution) -> Value {
        if self.obs.is_some() {
            self.obs.clone().unwrap()
        } else {
            dist.sample(rng)
        }
    }
}

/// Trait for handling choice and probability events during evaluation
pub trait ChoiceHandler {
    /// Called when a choice is made (sample or observe)
    /// Returns the final value and score
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String>;
}

/// Helper function to evaluate a distribution and either sample from it or score a value
pub fn eval_distribution(
    distribution_value: Value,
    obs: Option<&Value>,
    address: &Address,
    handler: &mut dyn ChoiceHandler,
) -> Result<Value, String> {
    match distribution_value {
        Value::Procedure(Procedure::Stochastic {
            name: dist_name,
            args,
        }) => {
            let args = args.unwrap_or_default();

            // Create choice event
            let event = ChoiceEvent {
                address: address.clone(),
                dist_name: dist_name.clone(),
                dist_args: args,
                obs: obs.cloned(),
            };

            // Handle the choice
            Ok(handler.on_choice(&event)?)
        }
        _ => Err("Distribution expression must yield a distribution".to_string()),
    }
}

/// Evaluates an expression against an environment, using the specified trace and random number generator.
/// The function is a monadic recursive evaluator, updating the trace with stochastic choices made during evaluation.
pub fn eval(
    expr: Expression,
    env: Rc<RefCell<Env>>,
    handler: &mut dyn ChoiceHandler,
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
            if let Some(value) = env.borrow().get(&name) {
                return Ok(value);
            }
            Err(format!("Unbound variable: {}", name))
        }

        Expression::List(exprs) => {
            // Evaluate each expression in the list
            let mut values = Vec::with_capacity(exprs.len());
            for e in &exprs {
                let val = eval(e.clone(), env.clone(), handler)?;
                values.push(val);
            }

            apply(values[0].clone(), values[1..].to_vec(), handler)
        }

        // Lambda creates closure
        Expression::Lambda(params, body) => Ok(Value::Procedure(Procedure::Lambda {
            params,
            body: body,
            env: env.clone(),
        })),

        // If evaluates the condition and executes the consequent or alternative
        Expression::If(cond, conseq, alt) => {
            let value = eval(*cond, env.clone(), handler)?;
            match value {
                Value::Boolean(true) => eval(*conseq, env.clone(), handler),
                Value::Boolean(false) => eval(*alt, env.clone(), handler),
                _ => Err("Condition must be a boolean".to_string()),
            }
        }

        // Define binds a name to a value in the current environment
        Expression::Define(name, expr) => {
            // Extend env by binding the value of (eval(expr, env)) to name
            // Return the extended env
            let value = eval(*expr, env.clone(), handler)?;
            env.borrow_mut().set(&name, value.clone());
            Ok(Value::Env(env))
        }

        // Quote returns the expression without evaluating it
        Expression::Quote(expr) => Ok(Value::Expr(*expr)),

        Expression::Sample { distribution, name } => {
            // Evaluate the name and distribution expressions
            let name = eval(*name, env.clone(), handler)?;
            let dist = eval(*distribution, env.clone(), handler)?;

            // Get the address of the name
            let sym = match name {
                Value::String(s) => s,
                Value::Procedure(_) | Value::List(_) | Value::Env(_) | Value::Expr(_) => {
                    return Err("sample: name must evaluate to a string".into())
                }
                other => format!("{:?}", other),
            };

            // Use the helper function to sample from the distribution with handler
            let addr = Address::Symbol(sym.clone());
            let value = eval_distribution(dist, None, &addr, handler)?;

            // Also bind to environment for consistent variable semantics
            env.borrow_mut().set(&sym, value.clone());

            Ok(value)
        }

        Expression::Observe {
            name,
            distribution,
            observed,
        } => {
            // Evaluate both the distribution and observed value
            let name = eval(*name, env.clone(), handler)?;
            let dist = eval(*distribution, env.clone(), handler)?;
            let value = eval(*observed, env.clone(), handler)?;

            let sym = match name {
                Value::String(s) => s,
                Value::Procedure(_) | Value::List(_) | Value::Env(_) | Value::Expr(_) => {
                    return Err("observe: name must evaluate to a string".into())
                }
                other => format!("{:?}", other),
            };

            // Use the helper function to score the observed value with handler
            let addr = Address::Symbol(sym.clone());
            let value = eval_distribution(dist, Some(&value), &addr, handler)?;

            // Also bind to environment for consistent variable semantics
            env.borrow_mut().set(&sym, value.clone());

            Ok(value)
        }

        Expression::ForEach { func, seq } => {
            // Evaluate the procedure expression once
            let proc = eval(*func.clone(), env.clone(), handler)?;

            // Evaluate the sequence expression once
            let seq = eval(*seq.clone(), env.clone(), handler)?;

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
                apply(proc.clone(), arg, handler)?;
            }

            // Return a Scheme-style "unit" – the empty list '()
            return Ok(Value::List(vec![]));
        }
    }
}

pub fn apply(
    func: Value,
    args: Vec<Value>,
    handler: &mut dyn ChoiceHandler,
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

            eval(*body, new_env, handler)
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

    env.borrow_mut().set(
        "bernoulli",
        Value::Procedure(Procedure::Stochastic {
            name: "bernoulli".to_string(),
            args: None,
        }),
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
