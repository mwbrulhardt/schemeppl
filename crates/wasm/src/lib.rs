use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use ppl::address::{Address, Selection};
use ppl::dsl::ast::Value;
use ppl::dsl::parser::parse_string;
use ppl::dsl::Literal;
use ppl::dynamic::trace::{Record, SchemeChoiceMap, SchemeDSLTrace, SchemeGenerativeFunction};
use ppl::gfi::{GenerativeFunction, Trace};
use ppl::inference::{metropolis_hastings, metropolis_hastings_with_proposal};
use rand::distributions::Distribution;

use ppl::dynamic::trace::make_extract_args;
use rand::{rngs::StdRng, SeedableRng};
use statrs::distribution::{Bernoulli, Normal};
use std::sync::{Arc, Mutex};

/// Custom error type for better JavaScript error handling
#[wasm_bindgen]
pub struct PplError {
    message: String,
}

#[wasm_bindgen]
impl PplError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl From<String> for PplError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

// Utility functions for better type conversion
fn js_value_to_value(js_val: &JsValue) -> Value {
    if let Some(num) = js_val.as_f64() {
        Value::Float(num)
    } else if let Some(s) = js_val.as_string() {
        Value::String(s)
    } else if let Some(b) = js_val.as_bool() {
        Value::Boolean(b)
    } else if js_val.is_null() || js_val.is_undefined() {
        Value::List(vec![]) // Use empty list for null/undefined
    } else {
        // Try to handle arrays
        if let Some(array) = js_val.dyn_ref::<js_sys::Array>() {
            let values: Vec<Value> = (0..array.length())
                .map(|i| js_value_to_value(&array.get(i)))
                .collect();
            Value::List(values)
        } else {
            Value::Float(0.0) // Default fallback
        }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// JavaScript-accessible RNG wrapper
#[wasm_bindgen]
pub struct JsRng {
    inner: Arc<Mutex<StdRng>>,
}

#[wasm_bindgen]
impl JsRng {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> JsRng {
        JsRng {
            inner: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }

    /// Re-seed the RNG with a new seed
    pub fn reseed(&self, seed: u64) {
        *self.inner.lock().unwrap() = StdRng::seed_from_u64(seed);
    }

    /// Generate a random f64 for testing purposes
    pub fn random(&self) -> f64 {
        use rand::Rng;
        self.inner.lock().unwrap().gen()
    }
}

#[wasm_bindgen]
pub struct JsGenerativeFunction {
    inner: SchemeGenerativeFunction,
}

#[wasm_bindgen]
impl JsGenerativeFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(param_names: Vec<String>, src: String) -> JsGenerativeFunction {
        // Parse the source code to get expressions
        let exprs = parse_string(&src);

        // Create the generative function with the data argument
        let gf = SchemeGenerativeFunction::new(exprs, param_names);

        JsGenerativeFunction { inner: gf }
    }

    /// Execute the generative function and return a trace
    pub fn simulate(&self, rng: &JsRng, args: &JsValue) -> Result<JsTrace, PplError> {
        let rust_args = if let Some(array) = args.dyn_ref::<Float64Array>() {
            vec![Value::List(
                array
                    .to_vec()
                    .into_iter()
                    .map(Value::Float)
                    .collect::<Vec<_>>(),
            )]
        } else {
            vec![js_value_to_value(args)]
        };

        let trace = self.inner.simulate(rng.inner.clone(), rust_args);
        Ok(JsTrace { inner: trace })
    }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct JsTrace {
    inner: SchemeDSLTrace,
}

#[wasm_bindgen]
impl JsTrace {
    /// Get the numeric value of a choice.
    pub fn get_choice(&self, name: &str) -> f64 {
        let addr = Address::Symbol(name.to_string());
        if let Some(value) = self.inner.get_choice_value(&addr) {
            match value {
                Value::Float(f) => f,
                Value::Integer(i) => i as f64,
                _ => 0.0,
            }
        } else {
            0.0
        }
    }

    /// Get the whole choice list as an object `{name: number, …}`.
    pub fn choices(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let choices = self.inner.get_choices();
        for (addr, record) in choices.iter() {
            let key = match &addr {
                Address::Symbol(s) => s.clone(),
                Address::Path(path) => path.join("/"),
            };

            let value = match record {
                Record::Choice(literal, _) => match literal {
                    Literal::Float(f) => *f,
                    Literal::Integer(i) => *i as f64,
                    Literal::Boolean(b) => {
                        if *b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                },
                _ => 0.0,
            };

            js_sys::Reflect::set(&obj, &JsValue::from_str(&key), &JsValue::from_f64(value))
                .unwrap();
        }
        obj.into()
    }

    /// Get the arguments used in the trace
    pub fn get_args(&self) -> JsValue {
        let args = self.inner.get_args();
        if let Some(Value::List(data)) = args.get(0) {
            let values: Vec<f64> = data
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Integer(i) => *i as f64,
                    _ => 0.0,
                })
                .collect();
            Float64Array::from(&values[..]).into()
        } else {
            Float64Array::new_with_length(0).into()
        }
    }

    /// Get the return value of the trace
    pub fn get_retval(&self) -> JsValue {
        if let Some(retval) = self.inner.get_retval() {
            match retval {
                Value::Float(f) => JsValue::from_f64(f),
                Value::Integer(i) => JsValue::from_f64(i as f64),
                Value::Boolean(b) => JsValue::from_bool(b),
                Value::String(s) => JsValue::from_str(&s),
                _ => JsValue::NULL,
            }
        } else {
            JsValue::NULL
        }
    }

    /// Get the data used in the trace (alias for get_args for backward compatibility)
    pub fn get_data(&self) -> Float64Array {
        if let Some(Value::List(data)) = self.inner.get_args().get(0) {
            let values: Vec<f64> = data
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Integer(i) => *i as f64,
                    _ => 0.0,
                })
                .collect();
            Float64Array::from(&values[..])
        } else {
            Float64Array::new_with_length(0)
        }
    }

    /// Un-normalised log-probability of the trace (handy for diagnostics).
    pub fn score(&self) -> f64 {
        self.inner.get_score()
    }
}

/// Metropolis-Hastings with the same signature as the inference module
#[wasm_bindgen]
pub fn metropolis_hastings_js(
    rng: &JsRng,
    trace: &JsTrace,
    selection: &js_sys::Array,
    check: bool,
    observations: Option<JsTrace>,
) -> Result<js_sys::Array, JsValue> {
    let sel_strings: Vec<String> = selection.iter().filter_map(|v| v.as_string()).collect();

    // For now, use Selection::All if we have selections
    let selection = if sel_strings.is_empty() {
        Selection::All
    } else {
        Selection::All // Simplified for now
    };

    // Convert observations if provided
    let observations = if let Some(obs) = observations {
        obs.inner.get_choices()
    } else {
        SchemeChoiceMap::new()
    };

    // Use the new method to get the concrete generative function
    let (next, accepted) = metropolis_hastings(
        rng.inner.clone(),
        trace.inner.clone(),
        trace.inner.get_gen_fn(),
        selection,
        check,
        observations,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Array::new();
    result.push(&JsTrace { inner: next }.into());
    result.push(&JsValue::from_bool(accepted));
    Ok(result)
}

/// Metropolis-Hastings with custom proposal function
#[wasm_bindgen]
pub fn metropolis_hastings_with_proposal_js(
    rng: &JsRng,
    trace: &JsTrace,
    proposal: &JsGenerativeFunction,
    proposal_args: &js_sys::Array,
    names: &js_sys::Array,
    check: bool,
    observations: Option<JsTrace>,
) -> Result<js_sys::Array, JsValue> {
    // Convert JS array to Vec<Value>
    let proposal_args: Vec<Value> = proposal_args
        .iter()
        .map(|js_val| {
            if let Some(num) = js_val.as_f64() {
                Value::Float(num)
            } else if let Some(s) = js_val.as_string() {
                Value::String(s)
            } else if let Some(b) = js_val.as_bool() {
                Value::Boolean(b)
            } else {
                Value::Float(0.0) // Default fallback
            }
        })
        .collect();

    // Convert observations if provided
    let observations = observations.map(|obs| obs.inner.get_choices());

    let names: Vec<Address> = names
        .iter()
        .filter_map(|v| {
            if let Some(name) = v.as_string() {
                Some(Address::Symbol(name.to_string()))
            } else {
                None
            }
        })
        .collect();

    let extract_args = make_extract_args(names);

    let observations = if let Some(obs) = observations {
        obs
    } else {
        SchemeChoiceMap::new()
    };

    let (next, accepted) = metropolis_hastings_with_proposal(
        rng.inner.clone(),
        trace.inner.clone(),
        &proposal.inner,
        proposal_args,
        &extract_args,
        check,
        observations,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Array::new();
    result.push(&JsTrace { inner: next }.into());
    result.push(&JsValue::from_bool(accepted));
    Ok(result)
}

// Keep the existing generate_data function unchanged
#[wasm_bindgen]
pub fn generate_data(
    mu1: f64,
    sigma1: f64,
    mu2: f64,
    sigma2: f64,
    p: f64,
    n: usize,
    rng: &JsRng,
) -> JsValue {
    let mut rng = rng.inner.lock().unwrap();

    let z_dist = Bernoulli::new(p).unwrap();
    let z: Vec<bool> = (0..n).map(|_| z_dist.sample(&mut *rng)).collect();

    let component1 = Normal::new(mu1, sigma1).unwrap();
    let c1: Vec<f64> = (0..n).map(|_| component1.sample(&mut *rng)).collect();

    let component2 = Normal::new(mu2, sigma2).unwrap();
    let c2: Vec<f64> = (0..n).map(|_| component2.sample(&mut *rng)).collect();

    let data: Vec<f64> = (0..n).map(|i| if z[i] { c1[i] } else { c2[i] }).collect();

    // Convert to numeric values for JavaScript (0 = false, 1 = true)
    let z_numeric: Vec<u8> = z.iter().map(|&x| if x { 1 } else { 0 }).collect();

    // Create a JavaScript object with data and labels
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("data"),
        &Float64Array::from(&data[..]).into(),
    )
    .unwrap();
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("labels"),
        &js_sys::Uint8Array::from(&z_numeric[..]).into(),
    )
    .unwrap();

    result.into()
}
