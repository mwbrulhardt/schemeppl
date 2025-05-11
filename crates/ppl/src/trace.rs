use std::fmt::Debug;
use std::hash::Hash;

use std::collections::HashMap;

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
