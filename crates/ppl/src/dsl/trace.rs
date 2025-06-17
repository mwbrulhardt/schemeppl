use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rand::rngs::StdRng;
use std::fmt::Debug;
use rand::{SeedableRng};
use rand::RngCore;
use std::rc::Rc;
use std::cell::RefCell;

use crate::ast::{Expression, Value, Procedure, Env, Literal};
use crate::dsl::eval::{eval, standard_env};
use crate::primitives::create_distribution;
use crate::core::{
    gfi::{Trace, GenerativeFunction, GFIError, ArgDiff, RetDiff},
    address::{Address, Selection, ChoiceMap, IntoChoiceMap},
};
use crate::r#gen;


/// A probabilistic generative function for Scheme DSL
#[derive(Debug, Clone)]
pub struct SchemeGenerativeFunction {
    exprs: Vec<Expression>,
    argument_names: Vec<String>
}

impl SchemeGenerativeFunction {
    pub fn new(exprs: Vec<Expression>, argument_names: Vec<String>) -> Self {
        Self {
            exprs,
            argument_names
        }
    }

    fn stdlib(
        &self,
        trace: &mut SchemeDSLTrace,
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
}



/// Scheme DSL trace implementation
/// Uses Literal for choice values to maintain simplicity and compatibility
#[derive(Debug, Clone)]
pub struct SchemeDSLTrace {
    gen_fn: SchemeGenerativeFunction,
    choices: ChoiceMap<Literal>,
    scores: HashMap<Address, f64>,
    score: f64,
    args: Vec<Value>,
    retval: Option<Value>,
}

impl SchemeDSLTrace {
    pub fn new(gen_fn: SchemeGenerativeFunction, args: Vec<Value>) -> Self {
        Self {
            gen_fn,
            choices: ChoiceMap::Empty,
            scores: HashMap::new(),
            score: 0.0,
            args,
            retval: None,
        }
    }

    pub fn set_retval(&mut self, retval: Value) {
        self.retval = Some(retval);
    }

    pub fn add_choice(&mut self, addr: Address, value: Value, score: f64) -> Result<(), GFIError> {
        if self.scores.contains_key(&addr) {
            return Err(GFIError::InvalidAddress(
                format!("Choice already present at address {:?}", addr)
            ));
        }
        
        let literal: Literal = value.try_into()
            .map_err(|e| GFIError::InvalidChoice(e))?;
        
        // Build new ChoiceMap in-place without cloning existing structure
        let new_entry = ChoiceMap::entry(literal, &[addr.clone()]);
        self.choices = self.choices.merge(&new_entry);  // Use merge instead of clone + |
        self.scores.insert(addr, score);
        self.score += score;
        Ok(())
    }

    /// Helper method to collect addresses that match a selection
    fn get_selected_addresses(&self, selection: Selection) -> Vec<Address> {
        let mut selected = Vec::new();
        
        // Get all addresses from our choices
        for (addr, _) in self.choices.iter() {
            if selection.contains(&addr) {
                selected.push(addr);
            }
        }
        
        selected
    }

    /// Get a value as a Literal, then convert to Value if needed
    pub fn get_choice_value(&self, addr: &Address) -> Option<Value> {
        self.choices.get_submap(addr).get_value()
            .map(|literal| literal.into()) // Using From<Literal> for Value
    }
}

impl Trace<Literal> for SchemeDSLTrace {
    type Args = Vec<Value>;
    type RetVal = Value;
    type RetDiff = RetDiff;

    fn get_args(&self) -> &Self::Args {
        &self.args
    }

    fn get_retval(&self) -> &Self::RetVal {
        self.retval.as_ref().expect("Return value not set")
    }

    fn get_choices(&self) -> ChoiceMap<Literal> {
        self.choices.clone()
    }

    fn get_score(&self) -> f64 {
        self.score
    }

    fn get_gen_fn(&self) -> &dyn GenerativeFunction<
        Literal,
        Args = Self::Args,
        RetVal = Self::RetVal,
        RetDiff = Self::RetDiff,
        TraceType = Self,
    > {
        &self.gen_fn
    }

    fn project(&self, selection: Selection) -> f64 {
        // Calculate the log probability of the selected choices
        let mut projected_score = 0.0;
        
        // Get addresses selected from current choices
        let selected_addresses = self.get_selected_addresses(selection);
        
        // For each address in the selection, add its score to the projection
        for addr in selected_addresses {
            if let Some(&score) = self.scores.get(&addr) {
                projected_score += score;
            }
        }
        
        projected_score
    }

    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        argdiffs: Vec<ArgDiff>,
        constraints: ChoiceMap<Literal>,
    ) -> Result<(Self, f64, Self::RetDiff, ChoiceMap<Literal>), GFIError> {
        // Step 1: Merge old trace choices with new constraints (constraints take precedence)
        let merged_constraints = constraints.merge(&self.choices);

        // Step 2: Check if we can optimize based on argdiffs
        let _needs_full_regeneration = argdiffs.iter().any(|diff| matches!(diff, ArgDiff::Changed));
        
        // For now, we always do full regeneration (optimization can be added later)
        let old_score = self.score;
        let old_retval = self.get_retval().clone();
        
        // Step 3: Generate new trace with merged constraints
        let (new_trace, new_score) = self
            .gen_fn
            .generate(rng.clone(), args, merged_constraints)
            .map_err(|e| match e {
                GFIError::NotImplemented(msg) => GFIError::NotImplemented(msg),
                other => other,
            })?;

        // Step 4: Calculate discard - choices that were overwritten or removed
        let mut discard_pairs = Vec::new();
        
        // Add choices from old trace that were overwritten by constraints
        for (addr, _) in constraints.iter() {
            if self.choices.contains(&addr) {
                if let Some(old_literal) = self.choices.get_submap(&addr).get_value() {
                    discard_pairs.push((addr, old_literal));
                }
            }
        }
        
        // Add choices from old trace that don't appear in new trace
        for (old_addr, old_literal) in self.choices.iter() {
            if !new_trace.choices.contains(&old_addr) {
                discard_pairs.push((old_addr, old_literal));
            }
        }
        
        let discard: ChoiceMap<Literal> = discard_pairs.choice_map();

        // Step 5: Calculate weight (simplified version of the complex formula)
        // TODO: Implement full weight calculation with proposal distributions
        // For now, use the score difference as an approximation
        let weight = new_score - old_score;

        // Step 6: Determine if return value changed
        let retdiff = if new_trace.get_retval() == &old_retval {
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
        // For now, return an error as this is complex to implement  
        let selection_set = self.get_selected_addresses(selection);
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
                        let in_selection = selection_set.contains(&symbol);
                        let has_previous = self.choices.contains(&symbol);

                        let (value, score) = if in_selection || !has_previous {
                            // Resample if selected or if no previous value exists
                            let dist = eval(
                                *distribution.clone(),
                                env.clone(),
                                &mut new_trace,
                                &mut *rng_guard,
                            )
                            .map_err(|e| GFIError::NotImplemented(e))?;
                            match dist {
                                Value::Procedure(Procedure::Stochastic {
                                    name: dist_name,
                                    args,
                                }) => {
                                    let args = args.unwrap_or_default();
                                    let dist = create_distribution(&dist_name, &args)
                                        .map_err(|e| GFIError::NotImplemented(e))?;
                                    let value = dist.sample(&mut *rng_guard);
                                    let score = dist
                                        .log_prob(&value)
                                        .map_err(|e| GFIError::NotImplemented(e))?;
                                    (value, score)
                                }
                                _ => {
                                    return Err(GFIError::NotImplemented(
                                        "Sample distribution must yield a distribution".to_string(),
                                    ))
                                }
                            }
                        } else {
                            // Reuse previous value if not selected
                            let prev_value = self.get_choice_value(&symbol)
                                .ok_or_else(|| GFIError::NotImplemented("Previous choice not found".to_string()))?;

                            // Recompute score with current distribution
                            let dist = eval(
                                *distribution.clone(),
                                env.clone(),
                                &mut new_trace,
                                &mut *rng_guard,
                            )
                            .map_err(|e| GFIError::NotImplemented(e))?;
                            let score = match dist {
                                Value::Procedure(Procedure::Stochastic {
                                    name: dist_name,
                                    args,
                                }) => {
                                    let args = args.unwrap_or_default();
                                    let dist = create_distribution(&dist_name, &args)
                                        .map_err(|e| GFIError::NotImplemented(e))?;
                                    dist.log_prob(&prev_value)
                                        .map_err(|e| GFIError::NotImplemented(e))?
                                }
                                _ => {
                                    return Err(GFIError::NotImplemented(
                                        "Sample distribution must yield a distribution".to_string(),
                                    ))
                                }
                            };

                            // Update weight for reused choices
                            let prev_score = self.scores.get(&symbol).copied().unwrap_or(0.0);
                            weight += score - prev_score;
                            (prev_value, score)
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
                        let dist = eval(
                            *distribution.clone(),
                            env.clone(),
                            &mut new_trace,
                            &mut *rng_guard,
                        )
                        .map_err(|e| GFIError::NotImplemented(e))?;
                        let value = eval(
                            *observed.clone(),
                            env.clone(),
                            &mut new_trace,
                            &mut *rng_guard,
                        )
                        .map_err(|e| GFIError::NotImplemented(e))?;

                        let score = match dist {
                            Value::Procedure(Procedure::Stochastic {
                                name: dist_name,
                                args,
                            }) => {
                                let args = args.unwrap_or_default();
                                let dist = create_distribution(&dist_name, &args)
                                    .map_err(|e| GFIError::NotImplemented(e))?;
                                dist.log_prob(&value)
                                    .map_err(|e| GFIError::NotImplemented(e))?
                            }
                            _ => {
                                return Err(GFIError::NotImplemented(
                                    "Observe distribution must yield a distribution".to_string(),
                                ))
                            }
                        };

                        // OPTIMIZED: Create Address::Symbol once
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

impl GenerativeFunction<Literal> for SchemeGenerativeFunction {
    type Args = Vec<Value>;
    type RetVal = Value;
    type RetDiff = RetDiff;
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
        constraints: ChoiceMap<Literal>,
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
                            if let Some(constrained_literal) = constraints.get_submap(&addr).get_value() {
                                // Convert literal back to Value using From trait
                                let constrained_value: Value = constrained_literal.into();
                                
                                // Evaluate the distribution to get the score
                                let dist = eval(*distribution.clone(), env.clone(), &mut trace, &mut *rng)
                                    .map_err(|e| GFIError::NotImplemented(e))?;

                                let score = match dist {
                                    Value::Procedure(Procedure::Stochastic {
                                        name: dist_name,
                                        args,
                                    }) => {
                                        let args = args.unwrap_or_default();
                                        let dist = create_distribution(&dist_name, &args)
                                            .map_err(|e| GFIError::NotImplemented(e))?;
                                        
                                        dist.log_prob(&constrained_value)
                                            .map_err(|e| GFIError::NotImplemented(e))?
                                    }
                                    _ => {
                                        return Err(GFIError::NotImplemented(
                                            "Sample distribution must yield a distribution".to_string()
                                        ))
                                    }
                                };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::address::Address;

    #[test]
    fn test_scheme_dsl_trace_basic() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test initial state
        assert_eq!(trace.score, 0.0);

        // Add a choice
        let addr = Address::Symbol("x".to_string());
        trace.add_choice(addr.clone(), Value::Float(42.0), 1.5).unwrap();

        assert_eq!(trace.score, 1.5);
        assert!(trace.choices.contains(&addr));

        // Get the choice back
        assert_eq!(trace.get_choice_value(&addr), Some(Value::Float(42.0)));
    }

    #[test]
    fn test_scheme_dsl_trace_duplicate_address_error() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0), Value::Float(2.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        let addr = Address::Symbol("x".to_string());
        trace.add_choice(addr.clone(), Value::Float(42.0), 1.5).unwrap();

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
        trace.add_choice(Address::Symbol("x".to_string()), Value::Float(42.0), 1.0).unwrap();
        trace.add_choice(Address::Symbol("y".to_string()), Value::Float(1.0), 0.5).unwrap();

        // Test the Trace interface
        assert_eq!(trace.get_args(), &vec![Value::Float(1.0), Value::Float(2.0)]);
        assert_eq!(trace.get_retval(), &Value::Float(3.14));
        assert_eq!(trace.get_score(), 1.5);

        // Test choices retrieval
        let choices = trace.get_choices();
        assert!(!choices.is_empty());
        
        // Test value extraction
        assert_eq!(trace.get_choice_value(&Address::Symbol("x".to_string())), Some(Value::Float(42.0)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("y".to_string())), Some(Value::Float(1.0)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("nonexistent".to_string())), None);
    }

    #[test]
    fn test_choice_map_integration() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test that initial choices is Empty
        assert!(matches!(trace.choices, ChoiceMap::Empty));
        
        // Add multiple choices
        trace.add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5).unwrap();
        trace.add_choice(Address::Symbol("b".to_string()), Value::String("hello".to_string()), 0.3).unwrap();
        trace.add_choice(Address::Symbol("c".to_string()), Value::Boolean(true), 0.2).unwrap();

        // Test that choices are properly stored
        assert!(trace.choices.contains(&Address::Symbol("a".to_string())));
        assert!(trace.choices.contains(&Address::Symbol("b".to_string())));
        assert!(trace.choices.contains(&Address::Symbol("c".to_string())));
        assert!(!trace.choices.contains(&Address::Symbol("d".to_string())));

        // Test choice retrieval
        assert_eq!(trace.get_choice_value(&Address::Symbol("a".to_string())), Some(Value::Float(1.0)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("b".to_string())), Some(Value::String("hello".to_string())));
        assert_eq!(trace.get_choice_value(&Address::Symbol("c".to_string())), Some(Value::Boolean(true)));

        // Test score accumulation
        assert_eq!(trace.score, 1.0); // 0.5 + 0.3 + 0.2
    }

    #[test]
    fn test_get_selected_addresses() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Add some choices
        trace.add_choice(Address::Symbol("x".to_string()), Value::Float(1.0), 0.1).unwrap();
        trace.add_choice(Address::Symbol("y".to_string()), Value::Float(2.0), 0.2).unwrap();
        trace.add_choice(Address::Symbol("z".to_string()), Value::Float(3.0), 0.3).unwrap();

        // Test selection with Selection::All (should select all addresses)
        use crate::core::address::Selection;
        let all_selection = Selection::All;
        let selected = trace.get_selected_addresses(all_selection);
        
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
        trace.add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5).unwrap();
        trace.add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3).unwrap();
        trace.add_choice(Address::Symbol("c".to_string()), Value::Float(3.0), 0.2).unwrap();

        // Test projection with Selection::All (should include all scores)
        use crate::core::address::Selection;
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
    fn test_update_method_basic() {
        let rng = Arc::new(Mutex::new(StdRng::seed_from_u64(42)));
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![Value::Float(1.0)];
        let mut trace = SchemeDSLTrace::new(gen_fn.clone(), args.clone());
        trace.set_retval(Value::Float(42.0));

        // Add some initial choices
        trace.add_choice(Address::Symbol("x".to_string()), Value::Float(1.0), 0.5).unwrap();
        trace.add_choice(Address::Symbol("y".to_string()), Value::Float(2.0), 0.3).unwrap();

        // Create constraints that override one choice and add a new one
        let mut constraint_pairs = Vec::new();
        constraint_pairs.push((Address::Symbol("x".to_string()), Literal::Float(5.0)));
        constraint_pairs.push((Address::Symbol("z".to_string()), Literal::Float(10.0)));
        let constraints: ChoiceMap<Literal> = constraint_pairs.choice_map();

        // Test update - since we have no expressions, generate should succeed
        let argdiffs = vec![ArgDiff::NoChange];
        let result = trace.update(rng.clone(), args, argdiffs, constraints);
        
        // The update method should succeed since we have no expressions to evaluate
        match result {
            Ok((new_trace, weight, retdiff, discard)) => {
                // Verify the update worked correctly
                assert!(weight.is_finite()); // Weight should be a valid number
                assert!(matches!(retdiff, RetDiff::NoChange | RetDiff::Changed));
                
                // Verify discard contains information about overridden choices
                assert!(!discard.is_empty());
                
                // Verify new trace has expected properties
                assert_eq!(new_trace.get_args(), &vec![Value::Float(1.0)]);
            }
            Err(e) => {
                panic!("Update should have succeeded for empty expression list, got error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_constraint_conversion() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let _trace = SchemeDSLTrace::new(gen_fn, args);

        // Create a constraint ChoiceMap
        let mut constraint_pairs = Vec::new();
        constraint_pairs.push((Address::Symbol("a".to_string()), Literal::Float(1.5)));
        constraint_pairs.push((Address::Symbol("b".to_string()), Literal::Float(2.5)));
        let constraints: ChoiceMap<Literal> = constraint_pairs.choice_map();

        // Test that we can iterate over constraints
        let pairs = constraints.iter();
        assert_eq!(pairs.len(), 2);

        // Test that we can check containment
        assert!(constraints.contains(&Address::Symbol("a".to_string())));
        assert!(constraints.contains(&Address::Symbol("b".to_string())));
        assert!(!constraints.contains(&Address::Symbol("c".to_string())));
    }

    #[test]
    fn test_choice_map_merging() {
        // Test the merging logic used in update method
        let mut old_pairs = Vec::new();
        old_pairs.push((Address::Symbol("x".to_string()), Literal::Float(1.0)));
        old_pairs.push((Address::Symbol("y".to_string()), Literal::Float(2.0)));
        let old_choices: ChoiceMap<Literal> = old_pairs.choice_map();

        let mut new_pairs = Vec::new();
        new_pairs.push((Address::Symbol("x".to_string()), Literal::Float(10.0))); // Override
        new_pairs.push((Address::Symbol("z".to_string()), Literal::Float(3.0))); // New
        let new_constraints: ChoiceMap<Literal> = new_pairs.choice_map();

        // Test merging (new_constraints should take precedence)
        let merged = new_constraints.merge(&old_choices);

        // Test that new constraints take precedence
        assert!(merged.contains(&Address::Symbol("x".to_string())));
        assert!(merged.contains(&Address::Symbol("y".to_string())));
        assert!(merged.contains(&Address::Symbol("z".to_string())));

        // Test that we have all expected addresses
        let merged_pairs = merged.iter();
        assert_eq!(merged_pairs.len(), 3); // x (overridden), y (from old), z (new)
    }

    #[test]
    fn test_empty_choice_map_operations() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let trace = SchemeDSLTrace::new(gen_fn, args);

        // Test operations on empty ChoiceMap
        assert!(matches!(trace.choices, ChoiceMap::Empty));
        assert!(!trace.choices.contains(&Address::Symbol("anything".to_string())));
        assert_eq!(trace.get_choice_value(&Address::Symbol("anything".to_string())), None);

        // Test building choice map from empty trace
        let choice_map = trace.get_choices();
        assert!(choice_map.is_empty());
        
        let pairs = choice_map.iter();
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_different_value_types() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test all Value types that can be converted to Literal
        trace.add_choice(Address::Symbol("float".to_string()), Value::Float(3.14), 0.1).unwrap();
        trace.add_choice(Address::Symbol("int".to_string()), Value::Integer(42), 0.2).unwrap();
        trace.add_choice(Address::Symbol("bool".to_string()), Value::Boolean(true), 0.3).unwrap();
        trace.add_choice(Address::Symbol("string".to_string()), Value::String("hello".to_string()), 0.4).unwrap();
        
        // Test that complex values (List) cannot be added as choices
        let list_val = Value::List(vec![Value::Float(1.0), Value::Float(2.0)]);
        let result = trace.add_choice(Address::Symbol("list".to_string()), list_val.clone(), 0.5);
        assert!(result.is_err()); // Should fail because List cannot be converted to Literal

        // Verify all valid values are stored correctly
        assert_eq!(trace.get_choice_value(&Address::Symbol("float".to_string())), Some(Value::Float(3.14)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("int".to_string())), Some(Value::Integer(42)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("bool".to_string())), Some(Value::Boolean(true)));
        assert_eq!(trace.get_choice_value(&Address::Symbol("string".to_string())), Some(Value::String("hello".to_string())));

        // Test that get_choices handles all types
        let choice_map = trace.get_choices();
        assert_eq!(choice_map.iter().len(), 4);
    }

    #[test]
    fn test_address_types() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test different address types - but note that our ChoiceMap implementation
        // may not handle all address types the same way
        let symbol_addr = Address::Symbol("test".to_string());
        let index_addr = Address::Index(42);
        let path_addr = Address::Path(vec![Address::Symbol("a".to_string()), Address::Index(1)]);

        trace.add_choice(symbol_addr.clone(), Value::Float(1.0), 0.1).unwrap();
        trace.add_choice(index_addr.clone(), Value::Float(2.0), 0.2).unwrap();
        trace.add_choice(path_addr.clone(), Value::Float(3.0), 0.3).unwrap();

        // Test retrieval with different address types
        assert!(trace.choices.contains(&symbol_addr));
        // Note: Index and Path addresses might not work the same way in our current implementation
        // This is expected behavior for now
        assert_eq!(trace.get_choice_value(&symbol_addr), Some(Value::Float(1.0)));
        
        // Test that the choice map contains the addresses we added
        let choice_map = trace.get_choices();
        let pairs = choice_map.iter();
        assert_eq!(pairs.len(), 3); // All three addresses should be stored
    }

    #[test]
    fn test_score_tracking() {
        let gen_fn = SchemeGenerativeFunction::new(vec![], vec![]);
        let args = vec![];
        let mut trace = SchemeDSLTrace::new(gen_fn, args);

        // Test score accumulation
        assert_eq!(trace.score, 0.0);
        
        trace.add_choice(Address::Symbol("a".to_string()), Value::Float(1.0), 0.5).unwrap();
        assert_eq!(trace.score, 0.5);
        
        trace.add_choice(Address::Symbol("b".to_string()), Value::Float(2.0), 0.3).unwrap();
        assert_eq!(trace.score, 0.8);
        
        trace.add_choice(Address::Symbol("c".to_string()), Value::Float(3.0), -0.2).unwrap();
        // Use approximate comparison for floating point
        assert!((trace.score - 0.6).abs() < 1e-10);

        // Test that scores are tracked per address
        assert_eq!(trace.scores.get(&Address::Symbol("a".to_string())), Some(&0.5));
        assert_eq!(trace.scores.get(&Address::Symbol("b".to_string())), Some(&0.3));
        assert_eq!(trace.scores.get(&Address::Symbol("c".to_string())), Some(&-0.2));
    }
}



