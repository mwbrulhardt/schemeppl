use crate::address::{Address, Selection};
use crate::dsl::ast::Value;
use crate::dsl::eval::{ChoiceEvent, ChoiceHandler};
use crate::dsl::primitives::create_distribution;
use crate::dynamic::trace::{Record, SchemeChoiceMap, SchemeDSLTrace};
use rand::rngs::StdRng;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

pub struct EmptyChoiceHandler;

impl ChoiceHandler for EmptyChoiceHandler {
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String> {
        Ok(event.obs.clone().unwrap())
    }
}

pub struct DefaultChoiceHandler {
    pub rng: Arc<Mutex<StdRng>>,
    pub trace: SchemeDSLTrace,
}

impl DefaultChoiceHandler {
    pub fn new(rng: Arc<Mutex<StdRng>>, trace: SchemeDSLTrace) -> Self {
        Self { rng, trace }
    }
}

impl ChoiceHandler for DefaultChoiceHandler {
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String> {
        let dist = create_distribution(&event.dist_name, &event.dist_args).unwrap();
        let rng = &mut *self.rng.lock().unwrap();

        // Generate the sample
        let sample = event.sample(rng, &dist);

        // Compute the score
        let score = dist.log_prob(&sample).unwrap();

        // Add to trace for inference purposes
        let _ = self
            .trace
            .add_choice(event.address.clone(), sample.clone(), score);

        Ok(sample)
    }
}

pub struct RegenerateHandler {
    pub rng: Arc<Mutex<StdRng>>,
    pub trace: SchemeDSLTrace,
    pub prev_trace: SchemeDSLTrace,
    pub weight: f64,
    pub selection: Selection,
    pub visited: HashSet<Address>,
}

impl RegenerateHandler {
    pub fn new(
        rng: Arc<Mutex<StdRng>>,
        trace: SchemeDSLTrace,
        prev_trace: SchemeDSLTrace,
        selection: Selection,
    ) -> Self {
        Self {
            rng,
            trace,
            prev_trace,
            weight: 0.0,
            selection,
            visited: HashSet::new(),
        }
    }
}

impl ChoiceHandler for RegenerateHandler {
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String> {
        let addr = &event.address;

        // Mark this address as visited if it hasn't been visited yet
        if !self.visited.contains(addr) {
            self.visited.insert(addr.clone());
        } else {
            return Err(format!("Address {} has already been visited", addr));
        }

        let dist = create_distribution(&event.dist_name, &event.dist_args).unwrap();

        // Check if this choice existed in the previous trace and is selected
        let has_previous = self.prev_trace.has_choice(addr);
        let is_selected = self.selection.contains(addr);

        // Generate the sample
        let rng = &mut *self.rng.lock().unwrap();
        let sample = if has_previous && is_selected {
            event.sample(rng, &dist)
        } else if has_previous && !is_selected {
            self.prev_trace.get_choice_value(addr).unwrap()
        } else {
            // New choice
            event.sample(rng, &dist)
        };

        // Compute the score
        let score = dist.log_prob(&sample)?;

        // Update the weight
        if has_previous && !is_selected {
            let prev_score = self.prev_trace.get_choice_score(addr).unwrap();

            self.weight += score - prev_score;
        }

        let _ = self.trace.add_choice(addr.clone(), sample.clone(), score);

        Ok(sample)
    }
}

pub struct GenerateHandler {
    pub rng: Arc<Mutex<StdRng>>,
    pub trace: SchemeDSLTrace,
    pub constraints: SchemeChoiceMap,
    pub weight: f64,
}

impl GenerateHandler {
    pub fn new(
        rng: Arc<Mutex<StdRng>>,
        trace: SchemeDSLTrace,
        constraints: SchemeChoiceMap,
    ) -> Self {
        Self {
            rng,
            weight: 0.0,
            trace,
            constraints,
        }
    }
}

impl ChoiceHandler for GenerateHandler {
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String> {
        let addr = event.address.clone();

        let dist = create_distribution(&event.dist_name, &event.dist_args)?;

        let is_constrained = self.constraints.contains_key(&addr);

        // Generate the sample
        let rng = &mut *self.rng.lock().unwrap();
        let sample = if is_constrained {
            self.constraints.get(&addr).unwrap().clone().into()
        } else {
            event.sample(rng, &dist)
        };

        // Compute the score
        let score = dist.log_prob(&sample)?;

        // Update the weight
        if is_constrained {
            self.weight += score;
        }

        let _ = self.trace.add_choice(addr, sample.clone(), score);

        Ok(sample)
    }
}

pub struct UpdateHandler {
    pub rng: Arc<Mutex<StdRng>>,
    pub trace: SchemeDSLTrace,
    pub prev_trace: SchemeDSLTrace,
    pub weight: f64,
    pub constraints: SchemeChoiceMap,
    pub visited: HashSet<Address>,
    pub discarded: SchemeChoiceMap,
}

impl UpdateHandler {
    pub fn new(
        rng: Arc<Mutex<StdRng>>,
        trace: SchemeDSLTrace,
        prev_trace: SchemeDSLTrace,
        constraints: SchemeChoiceMap,
    ) -> Self {
        Self {
            rng,
            weight: 0.0,
            trace,
            prev_trace,
            constraints,
            visited: HashSet::new(),
            discarded: SchemeChoiceMap::new(),
        }
    }
}

impl ChoiceHandler for UpdateHandler {
    fn on_choice(&mut self, event: &ChoiceEvent) -> Result<Value, String> {
        let addr = &event.address;
        let dist = create_distribution(&event.dist_name, &event.dist_args).unwrap();

        // Mark this address as visited if it hasn't been visited yet
        if !self.visited.contains(addr) {
            self.visited.insert(addr.clone());
        } else {
            return Err(format!("Address {} has already been visited", addr));
        }

        // Check if this choice existed in previous trace and is constrained
        let has_previous = self.prev_trace.has_choice(addr);
        let is_constrained = self.constraints.contains_key(addr);

        let prev_score = if has_previous {
            self.prev_trace.get_choice_score(addr).unwrap()
        } else {
            0.0
        };

        // Handle discarding for constrained choices that replace existing values
        if is_constrained && has_previous {
            let prev_value = self.prev_trace.get_choice_value(addr).unwrap();
            self.discarded.insert(
                addr.clone(),
                Record::Choice(prev_value.try_into().unwrap(), prev_score),
            );
        }

        // Generate the sample
        let rng = &mut *self.rng.lock().unwrap();
        let sample = if is_constrained {
            self.constraints.get(addr).unwrap().clone().into()
        } else if has_previous {
            self.prev_trace.get_choice_value(addr).unwrap()
        } else {
            event.sample(rng, &dist)
        };

        // Compute the score
        let score = dist.log_prob(&sample)?;

        // Update the weight
        if has_previous {
            self.weight += score - prev_score;
        } else if is_constrained {
            self.weight += score;
        }

        let _ = self.trace.add_choice(addr.clone(), sample.clone(), score);

        Ok(sample)
    }
}
