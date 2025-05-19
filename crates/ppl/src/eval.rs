use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;

use nalgebra::{Cholesky, DMatrix, DVector};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;

use crate::ast::{Env, Expression, HostFn, Literal, Procedure, Value};
use crate::primitives::create_distribution;
use crate::primitives::*;
use crate::r#gen;
use crate::trace::{ChoiceMap, Trace};

pub fn eval(
    expr: Expression,
    env: Rc<RefCell<Env>>,
    trace: &mut Trace<GenerativeFunction, String, Value>,
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
            if trace.choices.contains_key(&name) {
                // use a helper that returns the stored Value regardless of is_choice
                let val = trace.get_value(&name);
                return Ok(val);
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
            // Evaluate the expressions
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

            // Get the stochastic procedure and sample from it
            match dist {
                Value::Procedure(Procedure::Stochastic {
                    name: dist_name,
                    args,
                }) => {
                    let args = args.unwrap_or_default();
                    let dist = create_distribution(&dist_name, &args)?;
                    let value = dist.sample(rng);
                    let score = dist.log_prob(&value)?;

                    // Add to trace
                    trace.add_choice(addr.clone(), value.clone(), score);
                    Ok(value)
                }
                _ => Err("Sample distribution must yield a distribution".to_string()),
            }
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

            // Get the stochastic procedure and compute score
            let score = match dist {
                Value::Procedure(Procedure::Stochastic {
                    name: dist_name,
                    args,
                }) => {
                    let args = args.unwrap_or_default();
                    let dist = create_distribution(&dist_name, &args)?;
                    dist.log_prob(&value)?
                }
                _ => return Err("Observe distribution must yield a distribution".to_string()),
            };

            // Add to trace and return the observed value
            trace.add_choice(addr.clone(), value.clone(), score);
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
    trace: &mut Trace<GenerativeFunction, String, Value>,
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

/// A probabilistic *generative function* — the core abstraction of this PPL runtime.
///
/// The public interface (`simulate`, `generate`, `regenerate`, `propose`, and the
/// helper `rand`) is modelled after the generative-function interface of the
/// [Gen probabilistic programming system](https://github.com/probcomp/Gen.jl/blob/master/src/gen_fn_interface.jl).
/// Having the same conceptual API makes it easier to port inference code and
/// intuition from Gen.jl to this Rust implementation.
///
/// In short:
/// • `simulate`  — run the model forward and return a complete execution trace.
/// • `generate`  — run the model under soft constraints and return the trace and
///   its log-probability.
/// • `regenerate` — locally modify an existing trace by resampling a subset of
///   stochastic choices (mirroring Gen's `regenerate`).
/// • `propose`   — draw an initial trace together with an importance weight.
///
/// The struct stores the model body (`exprs`) as a vector of Scheme AST nodes,
/// the list of argument names expected by the model, per-choice scale
/// parameters used by proposal mechanisms, and an RNG.
///
/// See the linked Gen.jl source for the authoritative reference on the design
/// of these methods.
#[derive(Debug, Clone)]
pub struct GenerativeFunction {
    exprs: Vec<Expression>,
    argument_names: Vec<String>,
    scales: HashMap<String, f64>,
    rng: RefCell<StdRng>,
}

impl GenerativeFunction {
    pub fn new(
        exprs: Vec<Expression>,
        argument_names: Vec<String>,
        scales: HashMap<String, f64>,
        seed: u64,
    ) -> Self {
        Self {
            exprs,
            argument_names,
            scales,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    #[inline]
    pub fn rand(&self) -> std::cell::RefMut<'_, StdRng> {
        self.rng.borrow_mut()
    }

    fn stdlib(
        &self,
        trace: &mut Trace<GenerativeFunction, String, Value>,
        rng: &mut dyn RngCore,
    ) -> Rc<RefCell<Env>> {
        let env = standard_env();

        let lib = gen!(
            (define gensym (make-gensym))
            (define constrain (lambda (x) (observe (gensym) (condition #t) x)))
        );

        for expr in lib.iter() {
            let _ = eval(expr.clone(), env.clone(), trace, rng);
        }

        env
    }

    pub fn simulate(&self, args: Vec<Value>) -> Result<Trace<Self, String, Value>, String> {
        let mut trace = Trace::new(self.clone(), args.clone());

        {
            let mut rng = self.rand();

            let env = self.stdlib(&mut trace, &mut *rng);

            for (name, arg) in self.argument_names.iter().zip(args) {
                env.borrow_mut().set(name, arg.clone());
            }

            for expr in self.exprs.iter() {
                eval(expr.clone(), env.clone(), &mut trace, &mut *rng)?;
            }
        }

        Ok(trace)
    }

    pub fn generate(
        &self,
        args: Vec<Value>,
        constraints: HashMap<String, f64>,
    ) -> Result<(Trace<Self, String, Value>, f64), String> {
        let mut trace = Trace::new(self.clone(), args.clone());

        {
            let mut rng = self.rand();
            let env = self.stdlib(&mut trace, &mut *rng);

            // Set arguments in environment
            for (name, arg) in self.argument_names.iter().zip(args) {
                env.borrow_mut().set(name, arg.clone());
            }

            for expr in &self.exprs {
                match expr {
                    Expression::Sample { distribution, name } => {
                        let name = eval(*name.clone(), env.clone(), &mut trace, &mut *rng)?;

                        let addr = match name {
                            Value::String(s) => s,
                            Value::Procedure(_)
                            | Value::List(_)
                            | Value::Env(_)
                            | Value::Expr(_) => {
                                return Err("sample: name must evaluate to a string".into())
                            }
                            other => format!("{:?}", other),
                        };

                        if constraints.contains_key(&addr) {
                            // Evaluate the distribution
                            let dist =
                                eval(*distribution.clone(), env.clone(), &mut trace, &mut *rng)?;

                            let val = constraints[&addr];
                            let score = match dist {
                                Value::Procedure(Procedure::Stochastic {
                                    name: dist_name,
                                    args,
                                    ..
                                }) => {
                                    let args = args.unwrap_or_default();
                                    let dist = create_distribution(&dist_name, &args)?;
                                    dist.log_prob(&Value::Float(val))?
                                }
                                _ => {
                                    return Err(
                                        "Sample distribution must yield a distribution".to_string()
                                    )
                                }
                            };

                            trace.add_choice(addr.clone(), Value::Float(val), score);
                        } else {
                            eval(expr.clone(), env.clone(), &mut trace, &mut *rng)?;
                        }
                    }
                    _ => {
                        eval(expr.clone(), env.clone(), &mut trace, &mut *rng)?;
                    }
                }
            }
        }

        Ok((trace.clone(), trace.get_score()))
    }

    pub fn regenerate(
        &self,
        trace: Trace<Self, String, Value>,
        selection: &HashSet<String>,
    ) -> Result<(Trace<Self, String, Value>, f64), String> {
        let old_score = trace.get_score();
        let args = trace.get_args().clone();

        let mut keys: Vec<_> = selection.iter().cloned().collect();
        if keys.is_empty() {
            return Err("regenerate: empty selection".into());
        }
        keys.sort();

        let d = keys.len();

        let mut mean = DVector::zeros(d);
        for (i, k) in keys.iter().enumerate() {
            mean[i] = trace.get_choice(k).value.expect_float();
        }

        let mut sigma = DMatrix::<f64>::zeros(d, d);

        for (i, k) in keys.iter().enumerate() {
            sigma[(i, i)] = self.scales.get(k).copied().unwrap_or(1.0).powi(2);
        }
        for i in 0..d {
            for j in (i + 1)..d {
                let key = format!("{},{}", keys[i], keys[j]);
                if let Some(&v) = self.scales.get(&key) {
                    sigma[(i, j)] = v;
                    sigma[(j, i)] = v;
                }
            }
        }

        /* Cholesky factor L (Σ = L Lᵀ) */
        let l = Cholesky::new(sigma)
            .ok_or("regenerate: covariance not positive-definite")?
            .l();

        let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();

        let delta: DVector<f64> = {
            let mut rng = self.rand();
            let eps = DVector::from_fn(d, |_, _| normal.sample(&mut *rng));
            &l * eps
        };

        let proposals = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), mean[i] + delta[i]))
            .collect::<HashMap<_, _>>();

        let (new_trace, _) = self.generate(args.clone(), proposals)?;
        let weight = new_trace.get_score() - old_score;

        Ok((new_trace, weight))
    }

    pub fn propose(
        &self,
        args: Vec<Value>,
    ) -> Result<(ChoiceMap<String, Value>, f64, Value), String> {
        let trace = self.simulate(args)?;
        let weight = trace.get_score();
        Ok((trace.get_choices(), weight, trace.get_retval().clone()))
    }
}

pub fn mh(
    program: &GenerativeFunction,
    trace: Trace<GenerativeFunction, String, Value>,
    selection: &HashSet<String>,
) -> Result<(Trace<GenerativeFunction, String, Value>, bool), String> {
    let (updated, weight) = program.regenerate(trace.clone(), selection)?;
    let mut rng = program.rand();

    // Acceptance check
    if rng.gen::<f64>().ln() < weight {
        Ok((updated, true))
    } else {
        Ok((trace, false))
    }
}
