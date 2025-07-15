use rand::rngs::StdRng;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Index;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::dsl::ast::{Env, Expression, Literal, Value};
use crate::dsl::eval::{eval, standard_env};
use crate::dsl::handlers::{
    ChoiceHandler, DefaultChoiceHandler, EmptyChoiceHandler, GenerateHandler, RegenerateHandler,
    UpdateHandler,
};
use crate::{
    address::{Address, Selection},
    gfi::{GFIError, GenerativeFunction, Trace, Weight},
};

use std::collections::HashMap;

/// A record for choice map values
#[derive(Debug, Clone)]
pub enum Record {
    Choice(Literal, f64),
    Call(SchemeDSLTrace, f64),
}

impl Record {
    /// Extract the score from this record
    pub fn score(&self) -> f64 {
        match self {
            Record::Choice(_, score) => *score,
            Record::Call(_, score) => *score,
        }
    }
}

/// A simple choice map implementation using HashMap
#[derive(Debug, Clone, Default)]
pub struct SchemeChoiceMap {
    choices: HashMap<Address, Record>,
}

impl SchemeChoiceMap {
    pub fn new() -> Self {
        Self {
            choices: HashMap::new(),
        }
    }

    pub fn insert(&mut self, addr: Address, record: Record) {
        self.choices.insert(addr, record);
    }

    pub fn get(&self, addr: &Address) -> Option<&Record> {
        self.choices.get(addr)
    }

    pub fn contains_key(&self, addr: &Address) -> bool {
        self.choices.contains_key(addr)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Address, &Record)> {
        self.choices.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }

    /// Merge another choice map into this one
    pub fn merge_with(&mut self, other: &Self) {
        for (addr, record) in &other.choices {
            self.choices.insert(addr.clone(), record.clone());
        }
    }

    /// Filter choices by selection
    pub fn filter(&self, selection: &Selection) -> Self {
        let mut filtered = Self::new();
        for (addr, record) in &self.choices {
            if selection.contains(addr) {
                filtered.insert(addr.clone(), record.clone());
            }
        }
        filtered
    }
}

impl Index<Address> for SchemeChoiceMap {
    type Output = Record;

    fn index(&self, addr: Address) -> &Self::Output {
        &self.choices[&addr]
    }
}

/// A probabilistic generative function for Scheme DSL
#[derive(Debug, Clone)]
pub struct SchemeGenerativeFunction {
    exprs: Vec<Expression>,
    argument_names: Vec<String>,
}

impl SchemeGenerativeFunction {
    pub fn new(exprs: Vec<Expression>, argument_names: Vec<String>) -> Self {
        Self {
            exprs,
            argument_names,
        }
    }

    fn stdlib(&self) -> Rc<RefCell<Env>> {
        let env = standard_env();

        let lib = crate::gen!(
            (define gensym (make-gensym))
            (define constrain (lambda (x) (observe (gensym) (condition x) #t)))
        );

        let mut handler = EmptyChoiceHandler;

        for expr in lib.iter() {
            let _ = eval(expr.clone(), env.clone(), &mut handler);
        }

        env
    }

    /// Execute the program with a choice handler
    fn run(&self, args: &[Value], handler: &mut dyn ChoiceHandler) -> Result<Value, GFIError> {
        let env = self.stdlib();

        // Set arguments in environment
        for (name, arg) in self.argument_names.iter().zip(args) {
            env.borrow_mut().set(name, arg.clone());
        }

        // Execute each expression
        let mut last_value = Value::List(vec![]);
        for expr in &self.exprs {
            last_value = eval(expr.clone(), env.clone(), handler)
                .map_err(|e| GFIError::NotImplemented(e))?;
        }

        Ok(last_value)
    }
}

/// Scheme DSL trace implementation
#[derive(Debug, Clone)]
pub struct SchemeDSLTrace {
    gen_fn: SchemeGenerativeFunction,
    choices: SchemeChoiceMap,
    score: f64,
    args: Vec<Value>,
    retval: Option<Value>,
}

impl SchemeDSLTrace {
    pub fn new(gen_fn: SchemeGenerativeFunction, args: Vec<Value>) -> Self {
        Self {
            gen_fn,
            choices: SchemeChoiceMap::new(),
            score: 0.0,
            args,
            retval: None,
        }
    }

    pub fn set_retval(&mut self, retval: Value) {
        self.retval = Some(retval);
    }

    /// Get the concrete generative function (useful for WASM bindings)
    pub fn get_gen_fn(&self) -> &SchemeGenerativeFunction {
        &self.gen_fn
    }

    /// Add a choice to the trace
    pub fn add_choice(&mut self, addr: Address, value: Value, score: f64) -> Result<(), GFIError> {
        if self.choices.contains_key(&addr) {
            return Err(GFIError::InvalidAddress(format!(
                "Choice already present at address {:?}",
                addr
            )));
        }

        let literal: Literal = value.try_into().map_err(|e| GFIError::InvalidChoice(e))?;
        let record = Record::Choice(literal, score);
        self.choices.insert(addr, record);

        // Always add to total log probability (for assess/density computation)
        self.score += score;

        Ok(())
    }

    /// Get a value from the choice map
    pub fn get_choice_value(&self, addr: &Address) -> Option<Value> {
        if let Some(record) = self.choices.get(addr) {
            match record {
                Record::Choice(literal, _) => Some(literal.clone().into()),
                Record::Call(_, _) => None,
            }
        } else {
            None
        }
    }

    /// Get the score for a choice at a specific address
    pub fn get_choice_score(&self, addr: &Address) -> Option<f64> {
        if let Some(record) = self.choices.get(addr) {
            Some(record.score())
        } else {
            None
        }
    }

    /// Check if a choice already exists at this address (for constrained sampling)
    pub fn has_choice(&self, addr: &Address) -> bool {
        self.choices.contains_key(addr)
    }
}

impl Trace<SchemeChoiceMap, Option<Value>> for SchemeDSLTrace {
    type Args = Vec<Value>;

    fn get_args(&self) -> &Self::Args {
        &self.args
    }

    fn get_retval(&self) -> Option<Value> {
        self.retval.clone()
    }

    fn get_choices(&self) -> SchemeChoiceMap {
        self.choices.clone()
    }

    fn get_score(&self) -> f64 {
        self.score
    }

    fn get_gen_fn(
        &self,
    ) -> &dyn GenerativeFunction<SchemeChoiceMap, Option<Value>, Args = Self::Args, TraceType = Self>
    {
        &self.gen_fn
    }

    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        x: SchemeChoiceMap,
        args: Self::Args,
    ) -> Result<(Self, Weight, SchemeChoiceMap), GFIError>
    where
        Self: Sized,
    {
        // Delegate to the generative function's update implementation
        self.gen_fn.update(rng, self.clone(), Some(x), args)
    }
}

impl GenerativeFunction<SchemeChoiceMap, Option<Value>> for SchemeGenerativeFunction {
    type Args = Vec<Value>;
    type TraceType = SchemeDSLTrace;

    fn simulate(&self, rng: Arc<Mutex<StdRng>>, args: Self::Args) -> Self::TraceType {
        let mut handler =
            DefaultChoiceHandler::new(rng, SchemeDSLTrace::new(self.clone(), args.clone()));
        let last_value = self.run(&args, &mut handler).unwrap();
        let mut trace = handler.trace;
        trace.set_retval(last_value);
        trace
    }

    fn generate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        constraints: Option<SchemeChoiceMap>,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight), GFIError> {
        if constraints.is_none() {
            let trace = self.simulate(rng, args);
            let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 0.0);
            return Ok((trace, weight));
        }

        let mut handler = GenerateHandler::new(
            rng,
            SchemeDSLTrace::new(self.clone(), args.clone()),
            constraints.unwrap_or_default(),
        );

        let last_value = self.run(&args, &mut handler)?;
        handler.trace.set_retval(last_value);

        let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), handler.weight);
        Ok((handler.trace, weight))
    }

    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        trace: Self::TraceType,
        constraints: Option<SchemeChoiceMap>,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight, SchemeChoiceMap), GFIError>
    where
        Self: Sized,
    {
        let mut handler = UpdateHandler::new(
            rng,
            SchemeDSLTrace::new(self.clone(), args.clone()),
            trace.clone(),
            constraints.unwrap_or_default(),
        );

        let last_value = self.run(&args, &mut handler)?;
        handler.trace.set_retval(last_value);

        // Subtract scores of unvisited choices
        for (addr, record) in trace.get_choices().iter() {
            if !handler.visited.contains(addr) {
                handler.weight -= record.score();

                // Also add to discarded
                handler.discarded.insert(addr.clone(), record.clone());
            }
        }

        let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), handler.weight);
        Ok((handler.trace, weight, handler.discarded))
    }

    fn regenerate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        trace: Self::TraceType,
        selection: Selection,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight, Option<SchemeChoiceMap>), GFIError>
    where
        Self: Sized,
    {
        // Handle the case where no addresses are selected (weight = 0, trace unchanged)
        if matches!(selection, Selection::None) {
            let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 0.0);
            return Ok((trace, weight, None));
        }

        let mut handler = RegenerateHandler::new(
            rng,
            SchemeDSLTrace::new(self.clone(), args.clone()),
            trace.clone(),
            selection,
        );

        let last_value = self.run(&args, &mut handler)?;
        handler.trace.set_retval(last_value);

        // Subtract scores of unvisited choices from previous trace
        // This accounts for choices that were deleted during regeneration
        let mut discarded = SchemeChoiceMap::new();
        for (addr, record) in trace.get_choices().iter() {
            if !handler.visited.contains(addr) {
                handler.weight -= record.score();
                discarded.insert(addr.clone(), record.clone());
            }
        }

        let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), handler.weight);
        let discarded_opt = if discarded.is_empty() {
            None
        } else {
            Some(discarded)
        };

        Ok((handler.trace, weight, discarded_opt))
    }

    fn merge(
        &self,
        left: SchemeChoiceMap,
        right: SchemeChoiceMap,
        _check: Option<ndarray::ArrayD<f64>>,
    ) -> Result<(SchemeChoiceMap, Option<SchemeChoiceMap>), GFIError>
    where
        SchemeChoiceMap: Clone + Debug + Index<Address> + 'static,
        Self: Sized,
    {
        let mut merged = left.clone();
        let mut discarded = SchemeChoiceMap::new();

        for (addr, record) in right.iter() {
            if let Some(old_record) = merged.get(addr) {
                discarded.insert(addr.clone(), old_record.clone());
            }
            merged.insert(addr.clone(), record.clone());
        }

        let discard = if discarded.is_empty() {
            None
        } else {
            Some(discarded)
        };

        Ok((merged, discard))
    }

    fn filter(
        &self,
        x: SchemeChoiceMap,
        selection: Selection,
    ) -> Result<(Option<SchemeChoiceMap>, Option<SchemeChoiceMap>), GFIError>
    where
        SchemeChoiceMap: Clone + Debug + Index<Address> + 'static,
        Self: Sized,
    {
        let mut selected = SchemeChoiceMap::new();
        let mut unselected = SchemeChoiceMap::new();

        for (addr, record) in x.iter() {
            if selection.contains(addr) {
                selected.insert(addr.clone(), record.clone());
            } else {
                unselected.insert(addr.clone(), record.clone());
            }
        }

        let selected_opt = if selected.is_empty() {
            None
        } else {
            Some(selected)
        };
        let unselected_opt = if unselected.is_empty() {
            None
        } else {
            Some(unselected)
        };

        Ok((selected_opt, unselected_opt))
    }
}

// Add From conversion for Record to Value
impl From<Record> for Value {
    fn from(record: Record) -> Self {
        match record {
            Record::Choice(literal, _) => literal.into(),
            Record::Call(_, _) => Value::String("call".to_string()),
        }
    }
}

pub fn make_extract_args(
    names: Vec<Address>,
) -> Box<dyn Fn(&Vec<Value>, &SchemeDSLTrace) -> Result<Vec<Value>, String>> {
    Box::new(move |args: &Vec<Value>, trace: &SchemeDSLTrace| {
        let mut selected_args = args.clone();
        for addr in &names {
            if let Some(record) = trace.get_choices().get(addr) {
                selected_args.push(record.clone().into());
            }
        }
        Ok(selected_args)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::Address;
    use rand::SeedableRng;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_choice_map_operations() {
        let mut choice_map = SchemeChoiceMap::new();

        // Test empty map
        assert!(choice_map.is_empty());

        // Add values
        choice_map.insert(
            Address::Symbol("x".to_string()),
            Record::Choice(Literal::Float(1.0), 0.1),
        );
        choice_map.insert(
            Address::Symbol("y".to_string()),
            Record::Choice(Literal::String("hello".to_string()), 0.2),
        );

        // Test containment
        assert!(choice_map.contains_key(&Address::Symbol("x".to_string())));
        assert!(choice_map.contains_key(&Address::Symbol("y".to_string())));
        assert!(!choice_map.contains_key(&Address::Symbol("z".to_string())));

        // Test value retrieval
        assert!(choice_map.get(&Address::Symbol("x".to_string())).is_some());
        assert!(choice_map.get(&Address::Symbol("y".to_string())).is_some());
        assert!(choice_map.get(&Address::Symbol("z".to_string())).is_none());

        // Test iteration
        let pairs: Vec<_> = choice_map.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_scheme_dsl_trace_basic() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test initial state
        assert_eq!(trace.score, 0.0);

        // Add a choice
        let addr = Address::Symbol("x".to_string());
        trace
            .add_choice(addr.clone(), Value::Float(42.0), 1.5)
            .unwrap();

        assert_eq!(trace.score, 1.5);
        let choices = trace.get_choices();
        assert!(choices.contains_key(&addr));

        // Get the choice back
        assert_eq!(trace.get_choice_value(&addr), Some(Value::Float(42.0)));
    }

    #[test]
    fn test_scheme_dsl_trace_duplicate_address_error() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        let addr = Address::Symbol("x".to_string());
        trace
            .add_choice(addr.clone(), Value::Float(42.0), 1.5)
            .unwrap();

        // Try to add another value at the same address
        let result = trace.add_choice(addr, Value::Float(43.0), 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_interface() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);
        trace.set_retval(Value::Float(3.14));

        // Add some choices
        trace
            .add_choice(Address::Symbol("x".to_string()), Value::Float(42.0), 1.0)
            .unwrap();
        trace
            .add_choice(Address::Symbol("y".to_string()), Value::Float(1.0), 0.5)
            .unwrap();

        // Test the Trace interface
        assert_eq!(
            trace.get_args(),
            &vec![Value::Float(1.0), Value::Float(2.0)]
        );

        // get_retval() now returns Option<Value>
        let retval = trace.get_retval();
        assert!(retval.is_some());
        match retval.unwrap() {
            Value::Float(f) => assert_eq!(f, 3.14),
            _ => panic!("Expected float value"),
        }

        assert_eq!(trace.get_score(), 1.5);

        // Test choices retrieval
        let choices = trace.get_choices();
        assert!(!choices.is_empty());

        // Test value extraction
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("x".to_string())),
            Some(Value::Float(42.0))
        );
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("y".to_string())),
            Some(Value::Float(1.0))
        );
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("nonexistent".to_string())),
            None
        );
    }

    #[test]
    fn test_choice_map_integration() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test that initial choices is empty
        let initial_choices = trace.get_choices();
        assert!(initial_choices.is_empty());

        // Add multiple choices
        trace
            .add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5)
            .unwrap();
        trace
            .add_choice(
                Address::Symbol("b".to_string()),
                Value::String("hello".to_string()),
                0.3,
            )
            .unwrap();
        trace
            .add_choice(Address::Symbol("c".to_string()), Value::Boolean(true), 0.2)
            .unwrap();

        // Test that choices are properly stored
        let choices = trace.get_choices();
        assert!(choices.contains_key(&Address::Symbol("a".to_string())));
        assert!(choices.contains_key(&Address::Symbol("b".to_string())));
        assert!(choices.contains_key(&Address::Symbol("c".to_string())));
        assert!(!choices.contains_key(&Address::Symbol("d".to_string())));

        // Test choice retrieval
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("a".to_string())),
            Some(Value::Float(1.0))
        );
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("b".to_string())),
            Some(Value::String("hello".to_string()))
        );
        assert_eq!(
            trace.get_choice_value(&Address::Symbol("c".to_string())),
            Some(Value::Boolean(true))
        );

        // Test score accumulation
        assert_eq!(trace.score, 1.0); // 0.5 + 0.3 + 0.2
    }

    #[test]
    fn test_add_choice() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Add a choice that contributes to weight
        trace
            .add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5)
            .unwrap();
        assert_eq!(trace.score, 0.5);

        // Add a choice that does not contribute to weight
        trace
            .add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3)
            .unwrap();
        assert_eq!(trace.score, 0.8);

        // Add a choice that contributes to weight
        trace
            .add_choice(Address::Symbol("c".to_string()), Value::Float(3.0), 0.2)
            .unwrap();
        assert_eq!(trace.score, 1.0);

        // Add a duplicate choice
        let result = trace.add_choice(Address::Symbol("a".to_string()), Value::Float(4.0), 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_address_types() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test different address types - only Symbol and Path(Vec<String>) are valid
        let symbol_addr = Address::Symbol("test".to_string());
        let path_addr = Address::Path(vec!["a".to_string(), "b".to_string()]);

        trace
            .add_choice(symbol_addr.clone(), Value::Float(1.0), 0.1)
            .unwrap();
        trace
            .add_choice(path_addr.clone(), Value::Float(3.0), 0.3)
            .unwrap();

        // Test retrieval with different address types
        let choices = trace.get_choices();
        assert!(choices.contains_key(&symbol_addr));
        assert_eq!(
            trace.get_choice_value(&symbol_addr),
            Some(Value::Float(1.0))
        );

        // Test that the choice map contains the addresses we added
        let pairs: Vec<_> = choices.iter().collect();
        assert_eq!(pairs.len(), 2); // Both addresses should be stored
    }

    #[test]
    fn test_score_tracking() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test score accumulation
        assert_eq!(trace.score, 0.0);

        trace
            .add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5)
            .unwrap();
        assert_eq!(trace.score, 0.5);

        trace
            .add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3)
            .unwrap();
        assert_eq!(trace.score, 0.8);

        trace
            .add_choice(Address::Symbol("c".to_string()), Value::Float(3.0), -0.2)
            .unwrap();
        // Use approximate comparison for floating point
        assert!((trace.score - 0.6).abs() < 1e-10);

        // Test that scores are tracked per address
        assert_eq!(
            trace.get_choice_score(&Address::Symbol("a".to_string())),
            Some(0.5)
        );
        assert_eq!(
            trace.get_choice_score(&Address::Symbol("b".to_string())),
            Some(0.3)
        );
        assert_eq!(
            trace.get_choice_score(&Address::Symbol("c".to_string())),
            Some(-0.2)
        );
    }

    #[test]
    fn test_update_method() {
        // Create a simple generative function for testing
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];

        // Create initial trace with some choices
        let mut initial_trace = SchemeDSLTrace::new(gen_fn.clone(), args.clone());
        initial_trace
            .add_choice(Address::Symbol("x".to_string()), Value::Float(10.0), 0.5)
            .unwrap();
        initial_trace
            .add_choice(Address::Symbol("y".to_string()), Value::Float(20.0), 0.3)
            .unwrap();
        initial_trace.set_retval(Value::Float(42.0));

        // Create constraints (new values for some choices)
        let mut constraints = SchemeChoiceMap::new();
        constraints.insert(
            Address::Symbol("x".to_string()),
            Record::Choice(Literal::Float(15.0), 0.7), // New value for x
        );
        constraints.insert(
            Address::Symbol("z".to_string()),
            Record::Choice(Literal::Float(30.0), 0.4), // New choice z
        );

        // Call update method
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(42)));
        let result = initial_trace.update(rng, constraints.clone(), args.clone());

        // Verify the update succeeded
        assert!(result.is_ok());
        let (new_trace, _weight, discard) = result.unwrap();

        // Test 1: New trace should have the constrained values
        // x should have the new constrained value
        if let Some(x_value) = new_trace.get_choice_value(&Address::Symbol("x".to_string())) {
            assert_eq!(x_value, Value::Float(15.0));
        }

        // z should have the new constrained value
        if let Some(z_value) = new_trace.get_choice_value(&Address::Symbol("z".to_string())) {
            assert_eq!(z_value, Value::Float(30.0));
        }

        // Test 3: Discard should contain old values that were overwritten
        let has_old_x = discard.iter().any(|(addr, record)| {
            matches!(addr, Address::Symbol(s) if s == "x")
                && matches!(record, Record::Choice(Literal::Float(val), _) if *val == 10.0)
        });
        assert!(has_old_x, "Discard should contain the old value of x");

        // Test 5: Arguments should be preserved
        assert_eq!(new_trace.get_args(), &args);
    }
}
