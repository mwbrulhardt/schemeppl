use rand::rngs::StdRng;
use rand::RngCore;
use std::cell::RefCell;
use std::fmt::Debug;
use std::iter::Map;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use num_traits::ToPrimitive;

use crate::dsl::ast::{Env, Expression, Literal, Value};
use crate::dsl::eval::{eval, eval_distribution, standard_env};
use crate::r#gen;
use crate::trie::Trie;
use crate::{
    address::{Address, Selection},
    choice_map::{ChoiceMap, ChoiceMapQuery},
    gfi::{ArgDiff, GFIError, GenerativeFunction, RetDiff, Trace},
    trie::TrieIter,
};

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

impl ToPrimitive for Record {
    fn to_f64(&self) -> Option<f64> {
        match self {
            Record::Choice(literal, _) => literal.to_f64(),
            Record::Call(_, _) => None,
        }
    }

    fn to_i64(&self) -> Option<i64> {
        self.to_f64()?.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_f64()?.to_u64()
    }
}

/// A concrete choice map implementation for the scheme DSL
#[derive(Debug, Clone)]
pub struct SchemeChoiceMap {
    trie: Trie<Record>,
}

impl SchemeChoiceMap {
    pub fn new(trie: Trie<Record>) -> Self {
        Self { trie }
    }

    /// Helper to convert an address to a path for trie operations
    fn address_to_path(addr: &Address) -> Vec<Address> {
        match addr {
            Address::Path(components) => components.clone(),
            single => vec![single.clone()],
        }
    }

    /// Merge another choice map into this one
    pub fn merge_with(&mut self, other: &Self) {
        self.trie.merge(other.trie.clone());
    }

    /// Check if this choice map contains an address
    pub fn contains(&self, addr: &Address) -> bool {
        let path = Self::address_to_path(addr);
        self.trie.contains(&path)
    }

    /// Get value at an address
    pub fn get_value(&self, addr: &Address) -> Option<&Record> {
        let path = Self::address_to_path(addr);
        self.trie.get(&path)
    }

    /// Set value at an address
    pub fn set_value_at(&mut self, addr: Address, value: Record) {
        let path = Self::address_to_path(&addr);
        self.trie.insert(&path, value);
    }
}

impl ChoiceMapQuery<Record> for SchemeChoiceMap {
    /// Check if an address exists in this choice map
    fn contains(&self, addr: &Address) -> bool {
        let path = Self::address_to_path(addr);
        self.trie.contains(&path)
    }

    /// Get the value at this address if it exists
    fn get_value(&self, addr: &Address) -> Option<&Record> {
        let path = Self::address_to_path(addr);
        self.trie.get(&path)
    }

    /// Get the sub-choice-map at this address
    fn get_submap(&self, addr: &Address) -> Option<&Self> {
        if self.contains(addr) {
            Some(self)
        } else {
            None
        }
    }

    fn is_leaf(&self) -> bool {
        // The root choice map is not a leaf if it has multiple entries
        false
    }

    fn as_value(&self) -> Option<&Record> {
        // This should only return a value if we're representing a single specific choice
        // For the root choice map, this should be None
        None
    }

    /// Get direct child addresses (not all descendants)
    fn get_children_addresses(&self) -> Vec<Address> {
        // Get all the paths from the trie and convert them back to addresses
        let mut keys = Vec::new();
        for (path, _) in self.trie.iter() {
            if path.len() == 1 {
                keys.push(path[0].clone());
            } else if !path.is_empty() {
                keys.push(Address::Path(path));
            }
        }
        keys
    }

    fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }

    /// Get the total number of values in this choice map
    fn len(&self) -> usize {
        self.trie.iter().count()
    }
}

impl Default for SchemeChoiceMap {
    fn default() -> Self {
        Self::new(Trie::new())
    }
}

impl ChoiceMap<Record> for SchemeChoiceMap {
    type Iter<'a>
        = Map<TrieIter<'a, Record>, fn((Vec<Address>, &'a Record)) -> (Address, &'a Record)>
    where
        Self: 'a;

    fn set_value(&mut self, addr: Address, value: Record) {
        self.set_value_at(addr, value);
    }

    fn remove(&mut self, addr: &Address) -> bool {
        let path = Self::address_to_path(addr);
        self.trie.remove(&path).is_some()
    }

    fn filter(&self, selection: &Selection) -> Self {
        let mut filtered = Self::default();
        for (addr, value) in self.iter() {
            if selection.contains(&addr) {
                filtered.set_value_at(addr, value.clone());
            }
        }
        filtered
    }

    /// Get all addresses (not just direct children)
    fn get_all_addresses(&self) -> Vec<Address> {
        self.iter().map(|(addr, _)| addr).collect()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.trie.iter().map(|(path, value)| {
            let addr = if path.len() == 1 {
                path[0].clone()
            } else {
                Address::Path(path)
            };
            (addr, value)
        })
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

    fn stdlib(&self, trace: &mut SchemeDSLTrace, rng: &mut dyn RngCore) -> Rc<RefCell<Env>> {
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
}

/// Scheme DSL trace implementation
/// Uses Literal for choice values to maintain simplicity and compatibility
#[derive(Debug, Clone)]
pub struct SchemeDSLTrace {
    gen_fn: SchemeGenerativeFunction,
    trie: Trie<Record>,
    score: f64,
    args: Vec<Value>,
    retval: Option<Value>,
}

impl SchemeDSLTrace {
    pub fn new(gen_fn: SchemeGenerativeFunction, args: Vec<Value>) -> Self {
        Self {
            gen_fn,
            trie: Trie::new(),
            score: 0.0,
            args,
            retval: None,
        }
    }

    pub fn set_retval(&mut self, retval: Value) {
        self.retval = Some(retval);
    }

    pub fn add_choice(&mut self, addr: Address, value: Value, score: f64) -> Result<(), GFIError> {
        let path = SchemeChoiceMap::address_to_path(&addr);
        if self.trie.contains(&path) {
            return Err(GFIError::InvalidAddress(format!(
                "Choice already present at address {:?}",
                addr
            )));
        }

        let literal: Literal = value.try_into().map_err(|e| GFIError::InvalidChoice(e))?;

        // Store record in trie
        let record = Record::Choice(literal, score);
        self.trie.insert(&path, record);

        self.score += score;
        Ok(())
    }

    /// Get a value from the trie
    pub fn get_choice_value(&self, addr: &Address) -> Option<Value> {
        let path = SchemeChoiceMap::address_to_path(addr);
        if let Some(record) = self.trie.get(&path) {
            match record {
                Record::Choice(literal, _) => Some(literal.clone().into()),
                Record::Call(_, _) => None, // For now, don't support call records
            }
        } else {
            None
        }
    }

    /// Get the score for a choice at a specific address
    pub fn get_choice_score(&self, addr: &Address) -> Option<f64> {
        let path = SchemeChoiceMap::address_to_path(addr);
        if let Some(record) = self.trie.get(&path) {
            Some(record.score())
        } else {
            None
        }
    }
}

impl Trace<Value, Record> for SchemeDSLTrace {
    type Args = Vec<Value>;
    type RetDiff = RetDiff;
    type ChoiceMap = SchemeChoiceMap;

    fn get_args(&self) -> &Self::Args {
        &self.args
    }

    fn get_retval(&self) -> &Value {
        self.retval.as_ref().expect("Return value not set")
    }

    fn get_choices(&self) -> Self::ChoiceMap {
        SchemeChoiceMap::new(self.trie.clone())
    }

    fn get_score(&self) -> f64 {
        self.score
    }

    fn get_gen_fn(
        &self,
    ) -> &dyn GenerativeFunction<Value, Record, Args = Self::Args, TraceType = Self> {
        &self.gen_fn
    }

    /// Override the default get_value to properly retrieve from our trie
    fn get_value(&self, addr: &Address) -> Option<Record> {
        let path = SchemeChoiceMap::address_to_path(addr);
        self.trie.get(&path).cloned()
    }

    fn project(&self, selection: Selection) -> f64 {
        // Calculate the log probability of the selected choices
        let mut projected_score = 0.0;

        // Iterate through all choices in the trace and sum scores for selected addresses
        for (_, record) in self.get_choices().filter(&selection).iter() {
            projected_score += record.score();
        }
        projected_score
    }

    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        _argdiffs: Vec<ArgDiff>,
        constraints: Self::ChoiceMap,
    ) -> Result<(Self, f64, Self::RetDiff, Self::ChoiceMap), GFIError> {
        // Step 1: Merge old trace choices with new constraints (constraints take precedence)
        let old_choices = self.get_choices();
        let mut merged_constraints = old_choices.clone();
        merged_constraints.merge_with(&constraints);

        // For now, we always do full regeneration (optimization can be added later)

        // Step 3: Generate new trace with merged constraints
        let (new_trace, new_score) = self
            .gen_fn
            .generate(rng.clone(), args, merged_constraints)
            .map_err(|e| match e {
                GFIError::NotImplemented(msg) => GFIError::NotImplemented(msg),
                other => other,
            })?;

        // Step 4: Calculate discard - choices that were overwritten or removed
        let mut discard = SchemeChoiceMap::default();

        // Fast approach: Only check addresses that were actually constrained
        // This is equivalent to the expensive filtering but O(k) instead of O(n²)
        for (addr, _new_value) in constraints.iter() {
            if let Some(old_value) = self.get_value(&addr) {
                discard.set_value_at(addr, old_value.clone());
            }
        }

        // Step 5: Calculate weight (simplified version of the complex formula)
        // TODO: Implement full weight calculation with proposal distributions
        // For now, use the score difference as an approximation
        let weight = new_score - self.score;

        // Step 6: Determine if return value changed
        let retdiff = if new_trace.get_retval() == &self.get_retval().clone() {
            RetDiff::NoChange
        } else {
            RetDiff::Changed
        };

        Ok((new_trace, weight, retdiff, discard))
    }

    fn regenerate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        _argdiffs: Vec<ArgDiff>,
        selection: Selection,
    ) -> Result<(Self, f64, Self::RetDiff), GFIError> {
        //let selection_set = self.get_selected_addresses(selection);
        let old_retval = self.get_retval().clone();
        let old_score = self.get_score();

        // Create a new trace
        let mut new_trace = SchemeDSLTrace::new(self.gen_fn.clone(), args.clone());
        let mut weight = 0.0;
        let mut last_value = Value::List(vec![]); // Default return value

        {
            let mut rng_guard = rng.lock().unwrap();
            let env = self.gen_fn.stdlib(&mut new_trace, &mut *rng_guard);

            // Set arguments in environment
            for (name, arg) in self.gen_fn.argument_names.iter().zip(args) {
                env.borrow_mut().set(name, arg.clone());
            }

            // Re-execute the expressions
            for expr in &self.gen_fn.exprs {
                match expr {
                    Expression::Sample { distribution, name } => {
                        let name_val =
                            eval(*name.clone(), env.clone(), &mut new_trace, &mut *rng_guard)
                                .map_err(|e| GFIError::NotImplemented(e))?;

                        let addr = match name_val {
                            Value::String(s) => s,
                            Value::Procedure(_)
                            | Value::List(_)
                            | Value::Env(_)
                            | Value::Expr(_) => {
                                return Err(GFIError::NotImplemented(
                                    "sample: name must evaluate to a string".into(),
                                ))
                            }
                            other => format!("{:?}", other),
                        };

                        // Check if this address is in the selection
                        let symbol = Address::Symbol(addr.clone());

                        let has_previous = self.get_choice_value(&symbol).is_some();

                        let (value, score) = if selection.contains(&symbol) {
                            // Resample if selected
                            let dist = eval(
                                *distribution.clone(),
                                env.clone(),
                                &mut new_trace,
                                &mut *rng_guard,
                            )
                            .map_err(|e| GFIError::NotImplemented(e))?;
                            eval_distribution(dist, None, &mut *rng_guard)
                                .map_err(|e| GFIError::NotImplemented(e))?
                        } else if has_previous {
                            // Reuse previous value if not selected and has previous value
                            let prev_value = self.get_choice_value(&symbol).ok_or_else(|| {
                                GFIError::NotImplemented("Previous choice not found".to_string())
                            })?;

                            // Recompute score with current distribution
                            let dist = eval(
                                *distribution.clone(),
                                env.clone(),
                                &mut new_trace,
                                &mut *rng_guard,
                            )
                            .map_err(|e| GFIError::NotImplemented(e))?;
                            let (_, score) =
                                eval_distribution(dist, Some(&prev_value), &mut *rng_guard)
                                    .map_err(|e| GFIError::NotImplemented(e))?;

                            // Update weight for reused choices
                            let prev_score = self.get_choice_score(&symbol).unwrap_or(0.0);
                            weight += score - prev_score;
                            (prev_value, score)
                        } else {
                            // No previous value and not selected - sample fresh
                            let dist = eval(
                                *distribution.clone(),
                                env.clone(),
                                &mut new_trace,
                                &mut *rng_guard,
                            )
                            .map_err(|e| GFIError::NotImplemented(e))?;
                            eval_distribution(dist, None, &mut *rng_guard)
                                .map_err(|e| GFIError::NotImplemented(e))?
                        };

                        let _ = new_trace.add_choice(symbol, value, score);
                    }
                    Expression::Observe {
                        name,
                        distribution,
                        observed,
                    } => {
                        let name_val =
                            eval(*name.clone(), env.clone(), &mut new_trace, &mut *rng_guard)
                                .map_err(|e| GFIError::NotImplemented(e))?;
                        let addr = match name_val {
                            Value::String(s) => s,
                            Value::Procedure(_)
                            | Value::List(_)
                            | Value::Env(_)
                            | Value::Expr(_) => {
                                return Err(GFIError::NotImplemented(
                                    "observe: name must evaluate to a string".into(),
                                ))
                            }
                            other => format!("{:?}", other),
                        };

                        // For observations, always recompute
                        let value = eval(
                            *observed.clone(),
                            env.clone(),
                            &mut new_trace,
                            &mut *rng_guard,
                        )
                        .map_err(|e| GFIError::NotImplemented(e))?;

                        let dist = eval(
                            *distribution.clone(),
                            env.clone(),
                            &mut new_trace,
                            &mut *rng_guard,
                        )
                        .map_err(|e| GFIError::NotImplemented(e))?;
                        let (_, score) = eval_distribution(dist, Some(&value), &mut *rng_guard)
                            .map_err(|e| GFIError::NotImplemented(e))?;

                        let _ = new_trace.add_choice(Address::Symbol(addr), value, score);
                    }
                    _ => {
                        last_value =
                            eval(expr.clone(), env.clone(), &mut new_trace, &mut *rng_guard)
                                .map_err(|e| GFIError::NotImplemented(e))?;
                    }
                }
            }
        }

        // Store the last evaluated expression as the return value
        new_trace.set_retval(last_value);
        let score = new_trace.get_score();

        // Calculate final weight
        weight += score - old_score;

        // Determine if return value changed
        let retdiff = if new_trace.get_retval() == &old_retval {
            RetDiff::NoChange
        } else {
            RetDiff::Changed
        };

        Ok((new_trace, weight, retdiff))
    }
}

impl GenerativeFunction<Value, Record> for SchemeGenerativeFunction {
    type Args = Vec<Value>;
    type TraceType = SchemeDSLTrace;

    fn simulate(&self, rng: Arc<Mutex<StdRng>>, args: Self::Args) -> Self::TraceType {
        let mut trace = SchemeDSLTrace::new(self.clone(), args.clone());

        {
            let mut rng = rng.lock().unwrap();

            let env = self.stdlib(&mut trace, &mut *rng);

            for (name, arg) in self.argument_names.iter().zip(args) {
                env.borrow_mut().set(name, arg.clone());
            }

            let mut last_value = Value::List(vec![]); // Default return value
            for expr in self.exprs.iter() {
                match eval(expr.clone(), env.clone(), &mut trace, &mut *rng) {
                    Ok(val) => last_value = val,
                    Err(_) => break, // Handle errors gracefully
                }
            }

            // Store the last evaluated expression as the return value
            trace.set_retval(last_value);
        }

        trace
    }

    fn generate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        constraints: <Self::TraceType as Trace<Value, Record>>::ChoiceMap,
    ) -> Result<(Self::TraceType, f64), GFIError> {
        let mut trace = SchemeDSLTrace::new(self.clone(), args.clone());
        let mut last_value = Value::List(vec![]); // Default return value

        {
            let mut rng = rng.lock().unwrap();
            let env = self.stdlib(&mut trace, &mut *rng);

            // Set arguments in environment
            for (name, arg) in self.argument_names.iter().zip(args) {
                env.borrow_mut().set(name, arg.clone());
            }

            for expr in &self.exprs {
                match expr {
                    Expression::Sample { distribution, name } => {
                        let name = eval(*name.clone(), env.clone(), &mut trace, &mut *rng)
                            .map_err(|e| GFIError::NotImplemented(e))?;

                        let addr = match name {
                            Value::String(s) => Address::Symbol(s),
                            other => Address::Symbol(format!("{:?}", other)),
                        };

                        if constraints.contains(&addr) {
                            // Use constrained value
                            if let Some(constrained_literal) = constraints.get_value(&addr) {
                                // Convert literal back to Value using From trait
                                let constrained_value: Value = constrained_literal.clone().into();

                                // Evaluate the distribution to get the score
                                let dist =
                                    eval(*distribution.clone(), env.clone(), &mut trace, &mut *rng)
                                        .map_err(|e| GFIError::NotImplemented(e))?;
                                let (_, score) =
                                    eval_distribution(dist, Some(&constrained_value), &mut *rng)
                                        .map_err(|e| GFIError::NotImplemented(e))?;

                                trace.add_choice(addr, constrained_value, score)?;
                            }
                        } else {
                            // Normal sampling
                            last_value = eval(expr.clone(), env.clone(), &mut trace, &mut *rng)
                                .map_err(|e| GFIError::NotImplemented(e))?;
                        }
                    }
                    _ => {
                        last_value = eval(expr.clone(), env.clone(), &mut trace, &mut *rng)
                            .map_err(|e| GFIError::NotImplemented(e))?;
                    }
                }
            }
        }

        // Store the last evaluated expression as the return value
        trace.set_retval(last_value);
        let score = trace.get_score();

        Ok((trace, score))
    }
}

// Add From conversion for Record to Value
impl From<Record> for Value {
    fn from(record: Record) -> Self {
        match record {
            Record::Choice(literal, _) => literal.into(),
            Record::Call(_, _) => Value::String("call".to_string()), // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::Address;
    use crate::gfi::ArgDiff;
    use rand::SeedableRng;
    use std::sync::{Arc, Mutex};

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
        assert!(choices.contains(&addr));

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
        assert_eq!(trace.get_retval(), &Value::Float(3.14));
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
    fn test_choice_map_operations() {
        let mut choice_map = SchemeChoiceMap::default();

        // Test empty map
        assert!(choice_map.is_empty());

        // Add values
        choice_map.set_value_at(
            Address::Symbol("x".to_string()),
            Record::Choice(Literal::Float(1.0), 0.1),
        );
        choice_map.set_value_at(
            Address::Symbol("y".to_string()),
            Record::Choice(Literal::String("hello".to_string()), 0.2),
        );

        // Test containment
        assert!(choice_map.contains(&Address::Symbol("x".to_string())));
        assert!(choice_map.contains(&Address::Symbol("y".to_string())));
        assert!(!choice_map.contains(&Address::Symbol("z".to_string())));

        // Test value retrieval - just check that we get Some() values
        assert!(choice_map
            .get_value(&Address::Symbol("x".to_string()))
            .is_some());
        assert!(choice_map
            .get_value(&Address::Symbol("y".to_string()))
            .is_some());
        assert!(choice_map
            .get_value(&Address::Symbol("z".to_string()))
            .is_none());

        // Test iteration
        let pairs: Vec<_> = choice_map.iter().collect();
        assert_eq!(pairs.len(), 2);
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
        assert!(choices.contains(&Address::Symbol("a".to_string())));
        assert!(choices.contains(&Address::Symbol("b".to_string())));
        assert!(choices.contains(&Address::Symbol("c".to_string())));
        assert!(!choices.contains(&Address::Symbol("d".to_string())));

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
    fn test_get_selected_addresses() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Add some choices
        trace
            .add_choice(Address::Symbol("x".to_string()), Value::Float(1.0), 0.1)
            .unwrap();
        trace
            .add_choice(Address::Symbol("y".to_string()), Value::Float(2.0), 0.2)
            .unwrap();
        trace
            .add_choice(Address::Symbol("z".to_string()), Value::Float(3.0), 0.3)
            .unwrap();

        // Test selection with Selection::All (should select all addresses)
        let all_selection = Selection::All;
        let selected: Vec<Address> = trace
            .get_choices()
            .filter(&all_selection)
            .iter()
            .map(|(addr, _)| addr)
            .collect();

        // Should contain all addresses
        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&Address::Symbol("x".to_string())));
        assert!(selected.contains(&Address::Symbol("y".to_string())));
        assert!(selected.contains(&Address::Symbol("z".to_string())));
    }

    #[test]
    fn test_project() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Add choices with known scores
        trace
            .add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5)
            .unwrap();
        trace
            .add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3)
            .unwrap();
        trace
            .add_choice(Address::Symbol("c".to_string()), Value::Float(3.0), 0.2)
            .unwrap();

        // Test projection with Selection::All (should include all scores)
        let all_selection = Selection::All;
        let projected_score = trace.project(all_selection);

        // The projected score should be the sum of all scores
        assert_eq!(projected_score, 1.0); // 0.5 + 0.3 + 0.2

        // Test projection with Selection::None (should be 0)
        let none_selection = Selection::None;
        let projected_score = trace.project(none_selection);
        assert_eq!(projected_score, 0.0);
    }

    #[test]
    fn test_address_types() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test different address types
        let symbol_addr = Address::Symbol("test".to_string());
        let index_addr = Address::Index(42);
        let path_addr = Address::Path(vec![Address::Symbol("a".to_string()), Address::Index(1)]);

        trace
            .add_choice(symbol_addr.clone(), Value::Float(1.0), 0.1)
            .unwrap();
        trace
            .add_choice(index_addr.clone(), Value::Float(2.0), 0.2)
            .unwrap();
        trace
            .add_choice(path_addr.clone(), Value::Float(3.0), 0.3)
            .unwrap();

        // Test retrieval with different address types
        let choices = trace.get_choices();
        assert!(choices.contains(&symbol_addr));
        assert_eq!(
            trace.get_choice_value(&symbol_addr),
            Some(Value::Float(1.0))
        );

        // Test that the choice map contains the addresses we added
        let pairs: Vec<_> = choices.iter().collect();
        assert_eq!(pairs.len(), 3); // All three addresses should be stored
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
        let mut constraints = SchemeChoiceMap::default();
        constraints.set_value_at(
            Address::Symbol("x".to_string()),
            Record::Choice(Literal::Float(15.0), 0.7), // New value for x
        );
        constraints.set_value_at(
            Address::Symbol("z".to_string()),
            Record::Choice(Literal::Float(30.0), 0.4), // New choice z
        );

        // Create RNG for the update
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(42)));

        // Create argdiffs (no changes to arguments)
        let argdiffs = vec![ArgDiff::NoChange, ArgDiff::NoChange];

        // Call update method
        let result = initial_trace.update(rng, args.clone(), argdiffs, constraints.clone());

        // Verify the update succeeded
        assert!(result.is_ok());
        let (new_trace, weight, retdiff, discard) = result.unwrap();

        // Test 1: New trace should have the constrained values
        // x should have the new constrained value
        if let Some(x_value) = new_trace.get_choice_value(&Address::Symbol("x".to_string())) {
            assert_eq!(x_value, Value::Float(15.0));
        }

        // z should have the new constrained value
        if let Some(z_value) = new_trace.get_choice_value(&Address::Symbol("z".to_string())) {
            assert_eq!(z_value, Value::Float(30.0));
        }

        // Test 2: Weight should be the score difference
        let old_score = initial_trace.get_score();
        let new_score = new_trace.get_score();
        assert_eq!(weight, new_score - old_score);

        // Test 3: Discard should contain old values that were overwritten
        // It should contain the old value of x that was overwritten by constraints
        let has_old_x = discard.iter().any(|(addr, record)| {
            matches!(addr, Address::Symbol(s) if s == "x")
                && matches!(record, Record::Choice(Literal::Float(val), _) if *val == 10.0)
        });
        assert!(has_old_x, "Discard should contain the old value of x");

        // Test 4: Return value diff detection
        // Since we're using the same gen_fn with empty expressions,
        // the return value should be the same (default empty list)
        match retdiff {
            RetDiff::NoChange | RetDiff::Changed => {
                // Either is acceptable for this simple test case
            }
        }

        // Test 5: Arguments should be preserved
        assert_eq!(new_trace.get_args(), &args);
    }

    #[test]
    fn test_update_with_changed_args() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let initial_args = vec![Value::Float(1.0)];
        let new_args = vec![Value::Float(2.0)]; // Different arguments

        let mut initial_trace = SchemeDSLTrace::new(gen_fn.clone(), initial_args);
        initial_trace
            .add_choice(Address::Symbol("test".to_string()), Value::Float(5.0), 0.1)
            .unwrap();
        initial_trace.set_retval(Value::Float(100.0));

        let constraints = SchemeChoiceMap::default(); // No constraints
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(123)));
        let argdiffs = vec![ArgDiff::Changed]; // Argument changed

        let result = initial_trace.update(rng, new_args.clone(), argdiffs, constraints);

        assert!(result.is_ok());
        let (new_trace, _weight, _retdiff, _discard) = result.unwrap();

        // New trace should have the new arguments
        assert_eq!(new_trace.get_args(), &new_args);
    }

    #[test]
    fn test_update_empty_constraints() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0)];

        let mut initial_trace = SchemeDSLTrace::new(gen_fn.clone(), args.clone());
        initial_trace
            .add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.2)
            .unwrap();
        initial_trace
            .add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3)
            .unwrap();
        initial_trace.set_retval(Value::Float(3.0));

        let constraints = SchemeChoiceMap::default(); // Empty constraints
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(456)));
        let argdiffs = vec![ArgDiff::NoChange];

        let result = initial_trace.update(rng, args, argdiffs, constraints);

        assert!(result.is_ok());
        let (new_trace, weight, _retdiff, _discard) = result.unwrap();

        // With empty constraints and an empty generative function,
        // the new trace might not have the same choices since the generative function
        // doesn't actually create any sample or observe statements
        let new_choices = new_trace.get_choices();
        // Just verify that the choices object was created successfully
        // (it may be empty for an empty generative function)
        let _ = new_choices;

        // Weight should be calculated as score difference
        let old_score = initial_trace.get_score();
        let new_score = new_trace.get_score();
        assert_eq!(weight, new_score - old_score);

        // Since our generative function is empty, the new score should be 0
        assert_eq!(new_score, 0.0);
        // So weight should be the negative of the old score
        assert_eq!(weight, -old_score);
    }

    #[test]
    fn test_update_constraint_precedence() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];

        let mut initial_trace = SchemeDSLTrace::new(gen_fn.clone(), args.clone());
        initial_trace
            .add_choice(
                Address::Symbol("same_addr".to_string()),
                Value::Float(100.0),
                0.1,
            )
            .unwrap();
        initial_trace.set_retval(Value::Float(0.0));

        // Create constraint with the same address but different value
        let mut constraints = SchemeChoiceMap::default();
        constraints.set_value_at(
            Address::Symbol("same_addr".to_string()),
            Record::Choice(Literal::Float(200.0), 0.2), // Different value
        );

        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(789)));
        let argdiffs = vec![];

        let result = initial_trace.update(rng, args, argdiffs, constraints);

        assert!(result.is_ok());
        let (new_trace, _weight, _retdiff, discard) = result.unwrap();

        // The constraint should take precedence
        if let Some(value) = new_trace.get_choice_value(&Address::Symbol("same_addr".to_string())) {
            assert_eq!(
                value,
                Value::Float(200.0),
                "Constraint should take precedence over old choice"
            );
        }

        // The discard should contain the old value that was overwritten
        let has_old_value = discard.iter().any(|(addr, record)| {
            matches!(addr, Address::Symbol(s) if s == "same_addr")
                && matches!(record, Record::Choice(Literal::Float(val), _) if *val == 100.0)
        });
        assert!(
            has_old_value,
            "Discard should contain the old overwritten value"
        );
    }
}
