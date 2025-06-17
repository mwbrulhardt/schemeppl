use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use ppl::{mh, metropolis_hastings as mh_with_check, parse_string, GenerativeFunction, Trace, Value, ChoiceMap, ChoiceOrCallRecord};

use ppl::distributions::DistributionExtended;

use js_sys::{Object, Reflect};
use rand::{rngs::StdRng, SeedableRng};
use statrs::distribution::{Bernoulli, Normal};
use std::collections::{HashMap, HashSet};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn generate_data(
    mu1: f64,
    sigma1: f64,
    mu2: f64,
    sigma2: f64,
    p: f64,
    n: usize,
    seed: u64,
) -> JsValue {
    let mut rng = StdRng::seed_from_u64(seed);

    let z_dist = Bernoulli::new(p).unwrap();
    let z: Vec<bool> = (0..n).map(|_| z_dist.sample_dyn(&mut rng)).collect();

    let component1 = Normal::new(mu1, sigma1).unwrap();
    let c1: Vec<f64> = (0..n).map(|_| component1.sample_dyn(&mut rng)).collect();

    let component2 = Normal::new(mu2, sigma2).unwrap();
    let c2: Vec<f64> = (0..n).map(|_| component2.sample_dyn(&mut rng)).collect();

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

#[wasm_bindgen]
pub struct JsGenerativeFunction {
    inner: GenerativeFunction,
}

#[wasm_bindgen]
impl JsGenerativeFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(src: String, _scales: &Object, seed: u64) -> JsGenerativeFunction {
        let exprs = parse_string(&src);

        // Note: scales parameter is ignored since custom proposal logic was removed
        let gf = GenerativeFunction::new(exprs, vec!["data".into()], seed);

        JsGenerativeFunction { inner: gf }
    }

    pub fn simulate(&self, data: &Float64Array) -> Result<JsTrace, JsValue> {
        let wrapped = Value::List(
            data.to_vec()
                .into_iter()
                .map(Value::Float)
                .collect::<Vec<_>>(),
        );

        self.inner
            .simulate(vec![wrapped])
            .map(|tr| JsTrace { inner: tr })
            .map_err(|e| JsValue::from_str(&e))
    }

    #[wasm_bindgen]
    pub fn regenerate(
        &self,
        current: &JsTrace,
        selection: &js_sys::Array,
    ) -> Result<js_sys::Array, JsValue> {
        // convert JS array → HashSet<String>
        let sel: HashSet<String> = selection.iter().filter_map(|v| v.as_string()).collect();

        // call your Rust logic
        let (prop_trace, log_w) = self
            .inner
            .regenerate(current.inner.clone(), &sel)
            .map_err(|e| JsValue::from_str(&e))?;

        // pack result back for JS
        let out = js_sys::Array::new();
        out.push(&JsTrace { inner: prop_trace }.into());
        out.push(&JsValue::from_f64(log_w));
        Ok(out)
    }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct JsTrace {
    inner: Trace<GenerativeFunction, String, Value>,
}

#[wasm_bindgen]
impl JsTrace {
    /// Get the numeric value of a choice.
    pub fn get_choice(&self, name: &str) -> f64 {
        self.inner
            .get_choice(&name.to_string())
            .value
            .expect_float()
    }

    /// Get the whole choice list as an object `{name: number, …}`.
    pub fn choices(&self) -> JsValue {
        let obj = js_sys::Object::new();
        for (k, v) in self.inner.get_choices().choices.iter() {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str(k),
                &JsValue::from_f64(v.subtrace_or_retval.expect_float()),
            )
            .unwrap();
        }
        obj.into()
    }

    /// Get the data used in the trace.
    pub fn get_data(&self) -> Float64Array {
        if let Value::List(data) = &self.inner.get_args()[0] {
            let values: Vec<f64> = data.iter().map(|v| v.expect_float()).collect();
            Float64Array::from(&values[..])
        } else {
            Float64Array::new_with_length(0)
        }
    }

    /// Un-normalised log-probability of the trace (handy for diagnostics).
    pub fn score(&self) -> f64 {
        self.inner.score
    }
}

#[wasm_bindgen]
pub fn metropolis_hastings_simple(
    program: &JsGenerativeFunction,
    trace: &JsTrace,
    selection: &js_sys::Array,
) -> Result<js_sys::Array, JsValue> {
    let sel: HashSet<String> = selection.iter().filter_map(|v| v.as_string()).collect();
    let (next, accepted) = mh(&program.inner, trace.inner.clone(), &sel)
        .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Array::new();
    result.push(&JsTrace { inner: next }.into());
    result.push(&JsValue::from_bool(accepted));
    Ok(result)
}

/// Backward compatible alias for metropolis_hastings_simple
#[wasm_bindgen]
pub fn metropolis_hastings(
    program: &JsGenerativeFunction,
    trace: &JsTrace,
    selection: &js_sys::Array,
) -> Result<js_sys::Array, JsValue> {
    metropolis_hastings_simple(program, trace, selection)
}

#[wasm_bindgen]
pub fn metropolis_hastings_with_check(
    program: &JsGenerativeFunction,
    trace: &JsTrace,
    selection: &js_sys::Array,
    check: bool,
    observations: &Object,
) -> Result<js_sys::Array, JsValue> {
    let sel: HashSet<String> = selection.iter().filter_map(|v| v.as_string()).collect();
    
    // Convert observations Object to ChoiceMap
    let obs_map = if check {
        let mut choices = HashMap::new();
        let keys = js_sys::Object::keys(observations);
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().unwrap();
            let value = Reflect::get(observations, &key).unwrap();
            let value_f64 = value.as_f64().unwrap();
            
            // Create a choice record for the observation
            choices.insert(
                key_str,
                ChoiceOrCallRecord {
                    subtrace_or_retval: Value::Float(value_f64),
                    score: 0.0,
                    noise: f64::NAN,
                    is_choice: true,
                },
            );
        }
        ChoiceMap::new(choices)
    } else {
        ChoiceMap::new(HashMap::new())
    };

    let (next, accepted) = mh_with_check(
        trace.inner.clone(),
        &sel,
        check,
        &obs_map,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Array::new();
    result.push(&JsTrace { inner: next }.into());
    result.push(&JsValue::from_bool(accepted));
    Ok(result)
}
