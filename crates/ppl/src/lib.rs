use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::Hash;

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use rand::RngCore;
use statrs::distribution::Continuous;
use statrs::distribution::Discrete;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use std::ptr;
use std::convert::TryFrom;


// TODO: Fix the distribution primitives to be more unified.
// TODO: Break up the code into smaller files.
// TODO: Get WASM app working and depoloyed with example app for vercel.


/// Computes log-sum-exp of a slice of f64 values using the "log-sum-exp trick".
fn logsumexp(x: &[f64]) -> f64 {
    let mx = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = x.iter().map(|&lp| (lp - mx).exp()).sum();
    mx + sum_exp.ln()
}

pub trait DistributionExtended<T>: Debug {
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> T;
    fn log_prob(&self, value: T) -> f64;
    fn clone_box(&self) -> Box<dyn DistributionExtended<T>>;
}

// Blanket impl for all Clone + Debug types
impl<T, D> DistributionExtended<T> for D
where
    D: Debug + Clone + 'static + DistributionExtendedImpl<T>,
{
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> T {
        <Self as DistributionExtendedImpl<T>>::sample_dyn(self, rng)
    }
    fn log_prob(&self, value: T) -> f64 {
        <Self as DistributionExtendedImpl<T>>::log_prob(self, value)
    }
    fn clone_box(&self) -> Box<dyn DistributionExtended<T>> {
        Box::new(self.clone())
    }
}

// Helper trait for actual implementations
pub trait DistributionExtendedImpl<T>: Debug + Clone {
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> T;
    fn log_prob(&self, x: T) -> f64;
}

// Allow Box<dyn DistributionExtended<T>> to be cloned
impl<T> Clone for Box<dyn DistributionExtended<T>> {
    fn clone(&self) -> Box<dyn DistributionExtended<T>> {
        self.clone_box()
    }
}


impl DistributionExtendedImpl<bool> for statrs::distribution::Bernoulli {
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> bool {
        use rand::distributions::Distribution;
        self.sample(rng)
    }

    fn log_prob(&self, x: bool) -> f64 {
        self.ln_pmf(x.into())
    }
}


impl DistributionExtendedImpl<f64> for statrs::distribution::Normal {
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> f64 {
        use rand::distributions::Distribution;
        self.sample(rng)
    }
    fn log_prob(&self, value: f64) -> f64 {
        self.ln_pdf(value)
    }
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub flag: bool,
}

impl Condition {
    pub fn new(flag: bool) -> Self {
        Self {
            flag
        }
    }
}

impl DistributionExtended<bool> for Condition {
    fn sample_dyn(&self, _rng: &mut dyn RngCore) -> bool {
        true
    }
    fn log_prob(&self, v: bool) -> f64 {
        if v == self.flag {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn clone_box(&self) -> Box<dyn DistributionExtended<bool>> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Mixture<T> {
    pub log_weights: Vec<f64>,
    pub components: Vec<Box<dyn DistributionExtended<T>>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Debug + Clone + 'static> DistributionExtendedImpl<T> for Mixture<T> {
    fn sample_dyn(&self, rng: &mut dyn RngCore) -> T {
        let weights: Vec<f64> = self.log_weights.iter().map(|&lw| lw.exp()).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let idx = dist.sample(rng);
        self.components[idx].sample_dyn(rng)
    }

    fn log_prob(&self, value: T) -> f64 {
        let logps: Vec<f64> = self
            .components
            .iter()
            .zip(self.log_weights.iter())
            .map(|(comp, &logw)| logw + comp.log_prob(value.clone()))
            .collect();
        logsumexp(&logps)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceRecord<T> {
    pub value: T,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallRecord<T> {
    pub value: T,
    pub score: f64,
    pub noise: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceOrCallRecord<T> {
    pub subtrace_or_retval: T,
    pub score: f64,
    pub noise: f64,
    pub is_choice: bool,
}

pub fn choice_record<T: Clone + Debug>(value: &ChoiceOrCallRecord<T>) -> ChoiceRecord<T> {
    if !value.is_choice {
        panic!("Record must be a choice");
    }
    ChoiceRecord {
        value: value.subtrace_or_retval.clone(),
        score: value.score,
    }
}

pub fn call_record<T: Clone + Debug>(value: &ChoiceOrCallRecord<T>) -> CallRecord<T> {
    if value.is_choice {
        panic!("Record must be a call");
    }
    CallRecord {
        value: value.subtrace_or_retval.clone(),
        score: value.score,
        noise: value.noise,
    }
}

#[derive(Debug, Clone)]
pub struct ChoiceMap<K, T> {
    pub choices: HashMap<K, ChoiceOrCallRecord<T>>,
}

impl<K, T> ChoiceMap<K, T>
where
    K: Eq + Hash + Clone + Debug,
    T: Clone + Debug,
{
    pub fn new(choices: HashMap<K, ChoiceOrCallRecord<T>>) -> Self {
        ChoiceMap { choices }
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }

    pub fn get_choice(&self, addr: &K) -> T {
        let record = self.choices.get(addr).expect("No record at address");
        if !record.is_choice {
            panic!("Address is not a choice");
        }
        record.subtrace_or_retval.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Trace<T, K, V> {
    pub gen_fn: T,
    pub args: Vec<V>,
    pub choices: HashMap<K, ChoiceOrCallRecord<V>>,
    pub is_empty: bool,
    pub score: f64,
    pub noise: f64,
    pub value: Option<V>,
}

impl<T, K, V> Trace<T, K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Clone + Debug,
    T: Clone + Debug,
{
    pub fn new(gen_fn: T, args: Vec<V>) -> Self {
        Trace {
            gen_fn,
            args,
            choices: HashMap::new(),
            is_empty: true,
            score: 0.0,
            noise: 0.0,
            value: None,
        }
    }

    pub fn has_choice(&self, address: &K) -> bool {
        self.choices
            .get(address)
            .map(|r| r.is_choice)
            .unwrap_or(false)
    }

    pub fn has_call(&self, address: &K) -> bool {
        self.choices
            .get(address)
            .map(|r| !r.is_choice)
            .unwrap_or(false)
    }

    pub fn get_choice(&self, address: &K) -> ChoiceRecord<V> {
        let record = self.choices.get(address).expect("No record at address");
        choice_record(record)
    }

    pub fn get_call(&self, address: &K) -> CallRecord<V> {
        let record = self.choices.get(address).expect("No record at address");
        call_record(record)
    }

    pub fn add_choice(&mut self, address: K, retval: V, score: f64) {
        if self.choices.contains_key(&address) {
            panic!("Value or subtrace already present at address.");
        }
        self.choices.insert(
            address,
            ChoiceOrCallRecord {
                subtrace_or_retval: retval,
                score,
                noise: f64::NAN,
                is_choice: true,
            },
        );
        self.score += score;
        self.is_empty = false;
    }

    pub fn project(&self, selection: &std::collections::HashSet<K>) -> f64 {
        if selection.is_empty() {
            self.noise
        } else {
            panic!("Projection not implemented for non-empty selection.");
        }
    }

    pub fn add_call(&mut self, address: K, subtrace: Trace<T, K, V>) {
        if self.choices.contains_key(&address) {
            panic!("Value or subtrace already present at address.");
        }
        let score = subtrace.score;
        let noise = subtrace.project(&std::collections::HashSet::new());
        let submap = subtrace.get_choices();
        self.is_empty = self.is_empty && submap.is_empty();
        self.choices.insert(
            address,
            ChoiceOrCallRecord {
                subtrace_or_retval: subtrace.value.clone().unwrap(),
                score,
                noise,
                is_choice: false,
            },
        );
        self.score += score;
        self.noise += noise;
    }

    pub fn get_value(&self, address: &K) -> V {
        self.choices
            .get(address)
            .map(|r| r.subtrace_or_retval.clone())
            .expect("No value present at address.")
    }

    pub fn get_choices(&self) -> ChoiceMap<K, V> {
        if self.is_empty {
            ChoiceMap::new(HashMap::new())
        } else {
            ChoiceMap::new(self.choices.clone())
        }
    }

    pub fn get_args(&self) -> &Vec<V> {
        &self.args
    }

    pub fn get_retval(&self) -> &V {
        self.value.as_ref().expect("Trace has no return value.")
    }

    pub fn get_score(&self) -> f64 {
        self.score
    }

    pub fn get_gen_fn(&self) -> &T {
        &self.gen_fn
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Constant(Literal),
    Variable(String),
    List(Vec<Expression>),
    Lambda(Vec<String>, Box<Expression>),
    If(Box<Expression>, Box<Expression>, Box<Expression>),
    Define(String, Box<Expression>),
    Quote(Box<Expression>),
    Sample {
        distribution: Box<Expression>,
        name: String,
    },
    Observe {
        name: Box<Expression>,
        distribution: Box<Expression>,
        observed: Box<Expression>,
    },
    ForEach {
        func: Box<Expression>,
        seq: Box<Expression>
    }
}

               // <‑‑ drop the auto‑derives
pub struct Env {
    bindings: RefCell<HashMap<String, Value>>,
    ctr: AtomicUsize,
    parent: Option<Rc<RefCell<Env>>>,
}

// ---- manual Clone ----
impl Clone for Env {
    fn clone(&self) -> Self {
        Env {
            bindings: RefCell::new(self.bindings.borrow().clone()),
            // copy current counter value into a *new* atomic
            ctr: AtomicUsize::new(self.ctr.load(Ordering::Relaxed)),
            parent: self.parent.clone(),
        }
    }
}

// ---- manual PartialEq (optional—delete if you never compare Envs) ----
impl PartialEq for Env {
    fn eq(&self, other: &Self) -> bool {
        self.bindings.borrow().eq(&other.bindings.borrow())
            && self.parent == other.parent
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            bindings: RefCell::new(HashMap::new()),
            ctr: AtomicUsize::new(0),
            parent: None,
        }
    }

    pub fn with_parent(parent: Rc<RefCell<Env>>) -> Self {
        let ctr = parent.borrow().ctr.load(Ordering::Relaxed);
        Self {
            bindings: RefCell::new(HashMap::new()),
            ctr: AtomicUsize::new(ctr),
            parent: Some(parent),
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        self.bindings
            .borrow()
            .get(name)
            .cloned()
            .or_else(|| self.parent.as_ref()?.borrow().get(name))
    }

    pub fn set(&self, name: &str, val: Value) {
        self.bindings.borrow_mut().insert(name.to_string(), val);
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // show the symbols that are bound, and whether there is a parent
        let keys: Vec<_> = self.bindings.borrow().keys().cloned().collect();
        f.debug_struct("Env")
            .field("keys", &keys)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Procedure::Deterministic { .. } => f.write_str("<deterministic>"),
            Procedure::Stochastic { .. } => f.write_str("<stochastic>"),
            Procedure::Lambda { params, .. } => f
                .debug_struct("λ")
                .field("params", params)
                .finish_non_exhaustive(),
        }
    }
}

type HostFn = Rc<dyn Fn(Vec<Value>) -> Result<Value, String>>;

#[derive(Clone)]
pub enum Procedure {
    Lambda {
        params: Vec<String>,
        body: Box<Expression>,
        env: Rc<RefCell<Env>>,
    },
    Deterministic {
        func: HostFn,
    },
    Stochastic {
        args: Option<Vec<Value>>,
        sample: fn(Vec<Value>, &mut dyn RngCore) -> Result<Value, String>,
        log_prob: fn(Vec<Value>, Value) -> Result<f64, String>,
    },
}


impl PartialEq for Procedure {
    fn eq(&self, other: &Self) -> bool {
        use Procedure::*;

        match (self, other) {
            (Deterministic { func: f1 },
             Deterministic { func: f2 }) =>
                Rc::ptr_eq(f1, f2),

            (Lambda { params: p1, body: b1, env: e1 },
             Lambda { params: p2, body: b2, env: e2 }) =>
                p1 == p2 && b1 == b2 && Rc::ptr_eq(e1, e2),

            (Stochastic { args: a1, sample: s1, log_prob: l1 },
             Stochastic { args: a2, sample: s2, log_prob: l2 }) =>
                a1 == a2 &&
                ptr::eq(*s1 as *const (), *s2 as *const ()) &&
                ptr::eq(*l1 as *const (), *l2 as *const ()),

            _ => false,
        }
    }
}




#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<Value>),
    Procedure(Procedure),
    Expr(Expression),
    Env(Rc<RefCell<Env>>),
}


// -------------------------------------------------------------
// 1. Expect‑style helpers  (panics with a clear message)
// -------------------------------------------------------------
impl Value {
    pub fn expect_float(&self)   -> f64   { self.as_float().unwrap()   }
    pub fn expect_int(&self)     -> i64   { self.as_int().unwrap()     }
    pub fn expect_bool(&self)    -> bool  { self.as_bool().unwrap()    }
    pub fn expect_str(&self)     -> &str  { self.as_str().unwrap()     }

    // non‑panicking accessors return Option<…>
    pub fn as_float(&self) -> Option<f64> {
        match self { Value::Float(f) => Some(*f), Value::Integer(i) => Some(*i as f64), _ => None }
    }
    pub fn as_int(&self)   -> Option<i64> {
        match self { Value::Integer(i) => Some(*i), _ => None }
    }
    pub fn as_bool(&self)  -> Option<bool> {
        match self { Value::Boolean(b) => Some(*b), _ => None }
    }
    pub fn as_str(&self)   -> Option<&str> {
        match self { Value::String(s) => Some(s), _ => None }
    }
}


fn make_gensym(args: Vec<Value>) -> Result<Value, String> {
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

fn normal_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_normal_args(&args)?;
    Ok(Value::Float(dist.sample_dyn(rng)))
}

fn normal_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
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
        return Err("mixture: expected 2 arguments: list of distributions and list of weights".into());
    }
    // First argument: list of component specifications (each itself a list of normal parameters)
    let dist_specs = match &args[0] {
        Value::List(v) => v,
        _ => return Err("mixture: first argument must be a list of distribution parameter lists".into()),
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
    let mut components: Vec<Box<dyn DistributionExtended<f64>>> = Vec::with_capacity(dist_specs.len());
    for spec in dist_specs {
        let params = match spec {
            Value::Procedure(Procedure::Stochastic { args, .. }) => {
                args.clone().unwrap()
            },
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

fn mixture_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_mixture_args(&args)?;
    Ok(Value::Float(dist.sample_dyn(rng)))
}

fn mixture_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
    let dist = parse_mixture_args(&args)?;
    match value {
        Value::Float(f) => Ok(dist.log_prob(f)),
        Value::Integer(i) => Ok(dist.log_prob(i as f64)),
        _ => Err("mixture log_prob expects a numeric value".into()),
    }
}

fn parse_condition_args(args: &[Value]) -> Result<Box<dyn DistributionExtended<bool>>, String> {
    match args.len() {
        0 => {
            Ok(Box::new(Condition::new(true)))
        },
        1 => {
            let value = &args[0];

            match value {
                Value::Boolean(flag) => {
                    Ok(Box::new(Condition::new(*flag)))
                },
                _ => {
                    Err("condition expects a boolean paramter".into())
                }
                
            }
        },
        _ => {
            Err("condition only expects 1 parameter".into())
        }
        
    }
}

fn condition_sample(args: Vec<Value>, rng: &mut dyn RngCore) -> Result<Value, String> {
    let dist = parse_condition_args(&args)?;
    Ok(Value::Boolean(dist.sample_dyn(rng)))
}

fn condition_log_prob(args: Vec<Value>, value: Value) -> Result<f64, String> {
    let dist = parse_condition_args(&args)?;

    match value {
        Value::Boolean(x) => {
            Ok(dist.log_prob(x))
        }
        _ => {
            Err("Invalid input for domain boolean".into())
        }
    }
}

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
            // Evaluate the distribution
            let dist = eval(*distribution, env.clone(), trace, rng)?;

            // Get the stochastic procedure and sample from it
            match dist {
                Value::Procedure(Procedure::Stochastic {
                    args,
                    sample,
                    log_prob,
                }) => {
                    let args = args.unwrap_or_default();
                    let value = sample(args.clone(), rng)?;
                    let score = log_prob(args, value.clone())?;

                    // Add to trace
                    trace.add_choice(name.clone(), value.clone(), score);
                    Ok(value)
                }
                _ => Err("Sample distribution must yield a distribution".to_string()),
            }
        },

        Expression::Observe {
            name,
            distribution,
            observed
        } => {
            // Evaluate both the distribution and observed value
            let name = eval(*name, env.clone(), trace, rng)?;
            let dist = eval(*distribution, env.clone(), trace, rng)?;
            let value = eval(*observed, env.clone(), trace, rng)?;

            let addr = match name {
                Value::String(s) => s,
                Value::Procedure(_) | Value::List(_) | Value::Env(_) | Value::Expr(_)
                    => return Err("observe: name must evaluate to a string".into()),
                other => format!("{:?}", other),
            };

            // Get the stochastic procedure and compute score
            let score = match dist {
                Value::Procedure(Procedure::Stochastic {
                    args,
                    sample: _,
                    log_prob,
                }) => {
                    let args = args.unwrap_or_default();
                    log_prob(args, value.clone())?
                }
                _ => return Err("Observe distribution must yield a distribution".to_string()),
            };

            // Add to trace and return the observed value
            trace.add_choice(addr.clone(), value.clone(), score);
            Ok(value)
        },

        Expression::ForEach { 
            func,
            seq 
        } => {
            // Evaluate the procedure expression once
            let proc = eval(*func.clone(), env.clone(), trace, rng)?;

            // Evaluate the sequence expression once
            let seq  = eval(*seq.clone(),  env.clone(), trace, rng)?;

            let items = match seq {
                Value::List(v) => v,
                _ => return Err("for-each: second argument must be a list".into()),
            };

            // Call PROC on every element (spreading if the element itself is a list)
            for item in items {
                let arg = match item {
                    Value::List(v) => v,
                    v              => vec![v],
                };
                apply(proc.clone(), arg, trace, rng)?;
            }

            // Return a Scheme-style “unit” – the empty list '()
            return Ok(Value::List(vec![]));
        }

    }
}

fn apply(
    func: Value,
    args: Vec<Value>,
    trace: &mut Trace<GenerativeFunction, String, Value>,
    rng: &mut dyn RngCore,
) -> Result<Value, String> {
    match func {
        Value::Procedure(Procedure::Deterministic { func }) => func(args),
        Value::Procedure(Procedure::Stochastic {
            args: _,
            sample,
            log_prob,
        }) => Ok(Value::Procedure(Procedure::Stochastic {
            args: Some(args),
            sample,
            log_prob,
        })),
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
        Value::Procedure(Procedure::Deterministic { func: wrap(make_gensym) })
    );
    
    env.borrow_mut().set(
        "display",
        Value::Procedure(Procedure::Deterministic { func: wrap(display) })
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
        Value::Procedure(Procedure::Deterministic { func: wrap(eq) })
    );

    env.borrow_mut().set(
        "<", 
        Value::Procedure(Procedure::Deterministic { func: wrap(lt) })
    );

    // Data Structures
    env.borrow_mut().set(
        "list",
        Value::Procedure(Procedure::Deterministic { func: wrap(list) })
    );

    // Distribution Primitives
    env.borrow_mut().set(
        "normal",
        Value::Procedure(Procedure::Stochastic {
            args: None,
            sample: normal_sample,
            log_prob: normal_log_prob,
        }),
    );

    env.borrow_mut().set(
        "mixture", 
        Value::Procedure(Procedure::Stochastic { 
            args: None, 
            sample: mixture_sample, 
            log_prob: mixture_log_prob
        })
    );

    env.borrow_mut().set(
        "condition", 
    Value::Procedure(Procedure::Stochastic { 
            args: None, 
            sample: condition_sample, 
            log_prob: condition_log_prob 
        })
    );

    env
}

#[derive(Debug, Clone)]
pub struct GenerativeFunction {
    exprs: Vec<Expression>,
    argument_names: Vec<String>,
    scales: HashMap<String, f64>,
    seed: u64,
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
            seed,
        }
    }

    fn stdlib(&self, trace: &mut Trace<GenerativeFunction, String, Value>, rng: &mut dyn RngCore,) -> Rc<RefCell<Env>> {
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
        let mut rng = rand::thread_rng();
        let mut trace = Trace::new(self.clone(), args.clone());

        let env = self.stdlib(&mut trace, &mut rng);

        for (name, arg) in self.argument_names.iter().zip(args) {
            env.borrow_mut().set(name, arg.clone());
        }

        for expr in self.exprs.iter() {
            eval(expr.clone(), env.clone(), &mut trace, &mut rng)?;
        }

        Ok(trace)
    }

    pub fn generate(
        &self,
        args: Vec<Value>,
        constraints: HashMap<String, f64>,
    ) -> Result<(Trace<Self, String, Value>, f64), String> {
        let mut rng = rand::thread_rng();
        let mut trace = Trace::new(self.clone(), args.clone());
        let env = self.stdlib(&mut trace, &mut rng);

        // Set arguments in environment
        for (name, arg) in self.argument_names.iter().zip(args) {
            env.borrow_mut().set(name, arg.clone());
        }

        for expr in &self.exprs {
            match expr {
                Expression::Sample { distribution, name } => {
                    if constraints.contains_key(name) {
                        // Evaluate the distribution
                        let dist = eval(*distribution.clone(), env.clone(), &mut trace, &mut rng)?;

                        let val = constraints[name];
                        let score = match dist {
                            Value::Procedure(Procedure::Stochastic { args, log_prob, .. }) => {
                                let args = args.unwrap_or_default();
                                log_prob(args, Value::Float(val))?
                            }
                            _ => {
                                return Err(
                                    "Sample distribution must yield a distribution".to_string()
                                )
                            }
                        };

                        trace.add_choice(name.clone(), Value::Float(val), score);
                    } else {
                        eval(expr.clone(), env.clone(), &mut trace, &mut rng)?;
                    }
                }
                _ => {
                    eval(expr.clone(), env.clone(), &mut trace, &mut rng)?;
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

        let mut proposals = HashMap::new();
        for (addr, record) in trace.choices.iter() {
            if selection.contains(addr) {
                let Value::Float(old_val) = record.subtrace_or_retval else {
                    return Err(format!("Expected float at address {}", addr));
                };

                let Some(&scale) = self.scales.get(addr) else {
                    return Err(format!("Missing scale for {}", addr));
                };
                let new_val = statrs::distribution::Normal::new(old_val, scale)
                    .unwrap()
                    .sample(&mut rand::thread_rng());
                proposals.insert(addr.clone(), new_val);
            }
        }

        let (new_trace, _) = self.generate(args, proposals)?;

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
    program: GenerativeFunction,
    trace: Trace<GenerativeFunction, String, Value>,
    selection: HashSet<String>,
) -> Result<(Trace<GenerativeFunction, String, Value>, bool), String> {
    let (updated, weight) = program.regenerate(trace.clone(), &selection)?;
    let mut rng = rand::thread_rng();

    // Acceptance check
    if rng.gen::<f64>().ln() < weight {
        Ok((updated, true))
    } else {
        Ok((trace, false))
    }
}



impl TryFrom<lexpr::Value> for Expression {
    type Error = String;

    fn try_from(v: lexpr::Value) -> Result<Self, Self::Error> {
        use lexpr::Value::*;

        Ok(match v {
            Bool(b)   => Expression::Constant(Literal::Boolean(b)),

            Number(n) => {    
                if n.is_f64() {
                    Expression::Constant(Literal::Float(n.as_f64().unwrap()))
                }
                else {
                    Expression::Constant(Literal::Integer(n.as_i64().unwrap()))
                }
            },
            String(s) => Expression::Constant(Literal::String(s.into())),
            Symbol(s) => Expression::Variable(s.into()),

            Cons(pair) => {
                // Collect the list (and an eventual improper tail) into Vec<Expression>
                let mut elems = Vec::new();
                for (car, maybe_tail) in pair.into_iter() {
                    elems.push(Expression::try_from(car).unwrap());
                
                    if let Some(tail) = maybe_tail {
                        if !matches!(tail, lexpr::Value::Nil | lexpr::Value::Null) {
                            elems.push(Expression::try_from(tail).unwrap());
                        }
                    }
                }

                // special forms -----------------------------------------------------------
                match &elems[..] {
                    
                    // (quote x)
                    [Expression::Variable(k), x] if k == "quote" =>
                        Expression::Quote(Box::new(x.clone())),

                    // (lambda (args...) body...)
                    [Expression::Variable(k), Expression::List(arg_elems), body @ ..]
                        if k == "lambda" =>
                    {
                        let args = arg_elems.iter()
                            .map(|e| match e {
                                Expression::Variable(name) => Ok(name.clone()),
                                _ => Err("lambda arguments must be symbols")
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        let body_expr = if body.len() == 1 {
                            body[0].clone()
                        } else {
                            Expression::List(body.to_vec())
                        };
                        Expression::Lambda(args, Box::new(body_expr))
                    }

                    // (if test then else)
                    [Expression::Variable(k), test, then_, else_] if k == "if" =>
                        Expression::If(Box::new(test.clone()),
                                       Box::new(then_.clone()),
                                       Box::new(else_.clone())),

                    // (define name expr)
                    [Expression::Variable(k), Expression::Variable(name), expr]
                        if k == "define" => {
                            Expression::Define(name.clone(), Box::new(expr.clone()))
                        },

                    // (sample name dist)
                    [Expression::Variable(k), Expression::Variable(name), dist]
                        if k == "sample" => {
                            Expression::Sample { distribution: Box::new(dist.clone()), name: name.into() }
                        },

                    // (observe name dist value)
                    [Expression::Variable(k), name, dist, expr]
                        if k == "observe" => {
                            Expression::Observe { 
                                name: Box::new(name.clone()),
                                distribution: Box::new(dist.clone()), 
                                observed: Box::new(expr.clone())
                            }
                        },

                    // (for-each proc seq)
                    [Expression::Variable(k), proc, seq]
                        if k == "for-each" => {
                            Expression::ForEach { 
                                func: Box::new(proc.clone()), 
                                seq: Box::new(seq.clone()) 
                            }
                        }

                    // anything else → ordinary list
                    _ => Expression::List(elems),
                }
            }

            Nil | Null => Expression::List(vec![]),

            other => return Err(format!("unhandled lexpr value: {:?}", other)),
        })
    }
}


pub fn parse_string(input: &str) -> Vec<Expression> {
    // wrap in an extra parens so we get all top‐level forms at once:
    let v = lexpr::from_str(input)
        .unwrap_or_else(|e| panic!("lexpr parsing error: {}", e));
    if let lexpr::Value::Cons(pair) = v {
        // convert each car in turn
        let mut out = Vec::new();
        let mut rest = Some(( pair.car().clone(), pair.cdr().clone() ));
        while let Some((h, t)) = rest.take() {
            out.push(Expression::try_from(h).unwrap());
            rest = match t {
                lexpr::Value::Cons(pair2) => Some(( pair2.car().clone(), pair2.cdr().clone() )),
                lexpr::Value::Nil => None,
                lexpr::Value::Null => None,
                other => {
                    out.push(Expression::try_from(other).unwrap());
                    None
                }
            };
        }
        out
    } else {
        panic!("expected wrapped list");
    }
}


#[macro_export]
macro_rules! gen {
    ( $($tt:tt)* ) => {{
        // turn the tokens back into a string like "#t x (lambda (x) x) (* x x)"
        let src = stringify!($($tt)*);
        // wrap in parens so our parser sees one top-level list of forms
        let wrapped = format!("( {} )", src);
        // hand off to your parser
        let exprs = $crate::parse_string(&wrapped);

        exprs
    }};
}



#[cfg(test)]
mod tests {

    use super::*;
    use statrs::distribution::Normal;
    use rand::distributions::{Bernoulli, Distribution};

    #[test]
    fn test_normal() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let boxed_normal: Box<dyn DistributionExtended<f64>> = Box::new(normal);
        let cloned_normal = boxed_normal.clone();
        let mut rng = rand::thread_rng();
        let sample = cloned_normal.sample_dyn(&mut rng);
        let logp = cloned_normal.log_prob(sample);
        assert!(logp.is_finite());
    }

    #[test]
    fn test_condition() {
        let cond = Condition { flag: true };
        let mut rng = rand::thread_rng();
        let sample = cond.sample_dyn(&mut rng);
        assert_eq!(sample, true);
    }

    #[test]
    fn test_mixture() {
        let normal1 = Normal::new(0.0, 1.0).unwrap();
        let normal2 = Normal::new(5.0, 2.0).unwrap();

        // Box them as trait objects
        let boxed_normal1: Box<dyn DistributionExtended<f64>> = Box::new(normal1);
        let boxed_normal2: Box<dyn DistributionExtended<f64>> = Box::new(normal2);

        // Mixture of the two, with equal log-weights
        let mixture = Mixture {
            log_weights: vec![(0.5f64).ln(), (0.5f64).ln()],
            components: vec![boxed_normal1.clone(), boxed_normal2.clone()],
            _marker: std::marker::PhantomData,
        };

        // Box the mixture as a trait object
        let boxed_mixture: Box<dyn DistributionExtended<f64>> = Box::new(mixture);

        // Clone the mixture
        let cloned_mixture = boxed_mixture.clone();

        // Sample from the mixture
        let mut rng = rand::thread_rng();
        let sample = cloned_mixture.sample_dyn(&mut rng);

        // Compute log-probability of the sample
        let logp = cloned_mixture.log_prob(sample);

        assert!(logp.is_finite());
    }

    #[test]
    fn test_mh_model_0() {
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
    fn test_mh_model_1() {
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
        let c1: Vec<f64> = (0..num_samples).map(|_| component1.sample(&mut rng)).collect();

        let component2 = Normal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples).map(|_| component2.sample(&mut rng)).collect();

        let data: Vec<f64> = (0..num_samples).map(|i| {
            if z[i] {
                c1[i]
            }
            else {
                c2[i]
            }
        }).collect();

        for (i, x) in data.iter().enumerate() {
            println!("(observe x{:?} mix {:?})", i, x)
        }

        // Define model
        let mut model = vec![Expression::Define(
            "p".to_string(), 
            Box::new(Expression::Constant(Literal::Float(p)))
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

        model.push(
            Expression::Observe { 
                distribution: Box::new(Expression::List(vec![
                    Expression::Variable("condition".into()),
                    logic
                ])), 
                observed: Box::new(Expression::Constant(Literal::Boolean(true))), 
                name: Box::new(Expression::Constant(Literal::String("mu1-lt-mu2".to_string())))
            }
        );

        let mixture = Expression::List(vec![
            Expression::Variable("mixture".into()),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu1".into()),
                    Expression::Constant(Literal::Float(1.0))
                ]),
                Expression::List(vec![
                    Expression::Variable("normal".into()),
                    Expression::Variable("mu2".into()),
                    Expression::Constant(Literal::Float(1.0))
                ])
            ]),
            Expression::List(vec![
                Expression::Variable("list".into()),
                Expression::Variable("p".into()),
                Expression::List(vec![
                    Expression::Variable("-".into()),
                    Expression::Constant(Literal::Float(1.0)),
                    Expression::Variable("p".into())
                ])
            ]),
        ]);

        model.push(Expression::Define("mix".into(), Box::new(mixture)));

        for (i, &x) in data.iter().enumerate() {
            model.push(Expression::Observe {
                distribution: Box::new(Expression::Variable("mix".into())),//Box::new(mixture.clone()),
                observed: Box::new(Expression::Constant(Literal::Float(x))),
                name: Box::new(Expression::Constant(Literal::String(format!("x{}", i))))
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

        let variance_mu1: f64 =
            samples_mu1.iter().map(|x| (x - mean_mu1).powi(2)).sum::<f64>() / samples_mu1.len() as f64;

        println!("(Gaussian) mu1 mean: {:.3}, var: {:.3}", mean_mu1, variance_mu1);

        assert!((mean_mu1 + 1.0).abs() < 0.5);
        assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

        let mean_mu2: f64 = samples_mu2.iter().sum::<f64>() / samples_mu2.len() as f64;
        let variance_mu2: f64 =
            samples_mu2.iter().map(|x| (x - mean_mu2).powi(2)).sum::<f64>() / samples_mu2.len() as f64;

        println!("(Gaussian) mu2 mean: {:.3}, var: {:.3}", mean_mu2, variance_mu2);

        assert!((mean_mu2 - 1.0).abs() < 0.5);
        assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
    }


    #[test]
    fn test_model_compact() {
        
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
        let c1: Vec<f64> = (0..num_samples).map(|_| component1.sample(&mut rng)).collect();

        let component2 = Normal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples).map(|_| component2.sample(&mut rng)).collect();

        let data: Vec<f64> = (0..num_samples).map(|i| {
            if z[i] {
                c1[i]
            }
            else {
                c2[i]
            }
        }).collect();
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
        

        let mut scales = HashMap::new();
        scales.insert("mu1".to_string(), 1.0);
        scales.insert("mu2".to_string(), 1.0);

        let program = GenerativeFunction::new(model, vec!["data".to_string()], scales, 42);

        let mut trace = program.simulate(vec![wrapped_data]).unwrap();

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

        let mean_mu1: f64 = samples_mu1.iter().sum::<f64>() / samples_mu1.len() as f64;

        let variance_mu1: f64 =
            samples_mu1.iter().map(|x| (x - mean_mu1).powi(2)).sum::<f64>() / samples_mu1.len() as f64;

        println!("(Gaussian) mu1 mean: {:.3}, var: {:.3}", mean_mu1, variance_mu1);

        let mean_mu2: f64 = samples_mu2.iter().sum::<f64>() / samples_mu2.len() as f64;
        let variance_mu2: f64 =
            samples_mu2.iter().map(|x| (x - mean_mu2).powi(2)).sum::<f64>() / samples_mu2.len() as f64;

        println!("(Gaussian) mu2 mean: {:.3}, var: {:.3}", mean_mu2, variance_mu2);


        assert!((mean_mu1 + 1.0).abs() < 0.5);
        assert!(variance_mu1 > 0.0 && variance_mu1 < 2.0);

        assert!((mean_mu2 - 1.0).abs() < 0.5);
        assert!(variance_mu2 > 0.0 && variance_mu2 < 2.0);
    }



    #[test]
    fn test_model_compact_with_dsl_minimal() {
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
        let c1: Vec<f64> = (0..num_samples).map(|_| component1.sample(&mut rng)).collect();

        let component2 = Normal::new(mu2, sigma).unwrap();
        let c2: Vec<f64> = (0..num_samples).map(|_| component2.sample(&mut rng)).collect();

        let data: Vec<f64> = (0..num_samples).map(|i| {
            if z[i] {
                c1[i]
            }
            else {
                c2[i]
            }
        }).collect();
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
            // if accepted {
            //     history.push(new_trace.clone());
            // }
            history.push(new_trace.clone());
            trace = new_trace;
            num_accepted += accepted as u32;
        }

        println!("Acceptence Rate: {:.3}", num_accepted as f64 / n as f64);

        let mean_mu1: f64 = history.iter().map(|t| t.get_choice(&mu1_name).value.expect_float()).sum::<f64>() / history.len() as f64;
        let variance_mu1: f64 = history.iter().map(|t| (t.get_choice(&mu1_name).value.expect_float() - mean_mu1).powi(2)).sum::<f64>() / history.len() as f64;

        println!("(Gaussian) mu1 mean: {:.3}, var: {:.3}", mean_mu1, variance_mu1);

        let mean_mu2: f64 = history.iter().map(|t| t.get_choice(&mu2_name).value.expect_float()).sum::<f64>() / history.len() as f64;
        let variance_mu2: f64 = history.iter().map(|t| (t.get_choice(&mu2_name).value.expect_float() - mean_mu2).powi(2)).sum::<f64>() / history.len() as f64;

        println!("(Gaussian) mu2 mean: {:.3}, var: {:.3}", mean_mu2, variance_mu2);

    }
}
