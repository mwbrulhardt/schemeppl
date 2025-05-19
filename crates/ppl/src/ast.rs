use std::collections::HashMap;
use std::fmt::{self, Debug};

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        name: Box<Expression>,
    },
    Observe {
        name: Box<Expression>,
        distribution: Box<Expression>,
        observed: Box<Expression>,
    },
    ForEach {
        func: Box<Expression>,
        seq: Box<Expression>,
    },
}

pub struct Env {
    bindings: RefCell<HashMap<String, Value>>,
    ctr: AtomicUsize,
    parent: Option<Rc<RefCell<Env>>>,
}

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

impl PartialEq for Env {
    fn eq(&self, other: &Self) -> bool {
        self.bindings.borrow().eq(&other.bindings.borrow()) && self.parent == other.parent
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

pub type HostFn = Rc<dyn Fn(Vec<Value>) -> Result<Value, String>>;

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
        name: String,
        args: Option<Vec<Value>>,
    },
}

impl PartialEq for Procedure {
    fn eq(&self, other: &Self) -> bool {
        use Procedure::*;

        match (self, other) {
            (Deterministic { func: f1 }, Deterministic { func: f2 }) => Rc::ptr_eq(f1, f2),

            (
                Lambda {
                    params: p1,
                    body: b1,
                    env: e1,
                },
                Lambda {
                    params: p2,
                    body: b2,
                    env: e2,
                },
            ) => p1 == p2 && b1 == b2 && Rc::ptr_eq(e1, e2),

            (Stochastic { name: n1, args: a1 }, Stochastic { name: n2, args: a2 }) => {
                n1 == n2 && a1 == a2
            }

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

impl Value {
    pub fn expect_float(&self) -> f64 {
        self.as_float().unwrap()
    }
    pub fn expect_int(&self) -> i64 {
        self.as_int().unwrap()
    }
    pub fn expect_bool(&self) -> bool {
        self.as_bool().unwrap()
    }
    pub fn expect_str(&self) -> &str {
        self.as_str().unwrap()
    }

    // non‑panicking accessors return Option<…>
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}
