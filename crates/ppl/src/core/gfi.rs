use rand::rngs::StdRng;
use std::any::{TypeId, type_name};
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;

use crate::core::address::{Address, Selection, ChoiceMap};


// Error types
#[derive(Debug, Clone)]
pub enum GFIError {
    NotImplemented(String),
    InvalidChoice(String),
    InvalidAddress(String),
    ZeroProbability,
}


// Argument difference marker for updates
#[derive(Debug, Clone)]
pub enum ArgDiff {
    NoChange,
    Changed,
}

// Return value difference marker
#[derive(Debug, Clone)]
pub enum RetDiff {
    NoChange,
    Changed,
}

/// Abstract trait for a trace of a generative function.
/// Generic over the choice value type C to allow flexibility in value representation.
pub trait Trace<C>: Debug 
where 
    C: Clone + Debug + 'static,
{
    type Args: Clone + Debug + 'static;
    type RetVal: Clone + Debug + 'static;
    type RetDiff: Clone + Debug + 'static;

    /// Return the argument tuple for a given execution.
    fn get_args(&self) -> &Self::Args;

    /// Return the return value of the given execution.
    fn get_retval(&self) -> &Self::RetVal;

    /// Get the choice map for this trace
    /// Returns owned values to enable on-demand construction and caching
    fn get_choices(&self) -> ChoiceMap<C>;

    /// Return log(p(r, t; x) / q(r; x, t))
    ///
    /// When there is no non-addressed randomness, this simplifies to the log probability log p(t; x).
    fn get_score(&self) -> f64;

    /// Return the generative function that produced the given trace.
    fn get_gen_fn(
        &self,
    ) -> &dyn GenerativeFunction<
        C,
        Args = Self::Args,
        RetVal = Self::RetVal,
        RetDiff = Self::RetDiff,
        TraceType = Self,
    >;

    /// Get the value of the random choice at address `addr`.
    fn get_value(&self, addr: &Address) -> Option<C> {
        let choices = self.get_choices();
        if choices.contains(addr) {
            choices.get_submap(addr).get_value()
        } else {
            None
        }
    }

    /// Estimate the probability that the selected choices take the values they do in a trace.
    ///
    /// Given a trace (x, r, t) and a set of addresses A (selection),
    /// let u denote the restriction of t to A. Return the weight: log(p(r, t; x) / (q(t; u, x) * q(r; x, t)))
    fn project(&self, selection: Selection) -> f64;

    /// Update a trace by changing the arguments and/or providing new values for some choices.
    ///
    /// Given a previous trace (x, r, t), new arguments x', and a map u (constraints),
    /// return a new trace (x', r', t') that is consistent with u.
    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        argdiffs: Vec<ArgDiff>,
        constraints: ChoiceMap<C>,
    ) -> Result<(Self, f64, Self::RetDiff, ChoiceMap<C>), GFIError>
    where
        Self: Sized;

    /// Update a trace by randomly sampling new values for selected random choices.
    ///
    /// Given a previous trace (x, r, t), new arguments x', and a set of addresses A (selection),
    /// return a new trace (x', t') such that t' agrees with t on all addresses not in A.
    fn regenerate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        argdiffs: Vec<ArgDiff>,
        selection: Selection,
    ) -> Result<(Self, f64, Self::RetDiff), GFIError>
    where
        Self: Sized;
}

/// Abstract trait for a generative function.
/// Generic over the choice value type C to allow flexibility in value representation.
///
/// Corresponds to Julia's GenerativeFunction{T,U <: Trace} where:
/// - T is the return value type (RetVal)
/// - U is the trace type (TraceType)
pub trait GenerativeFunction<C>: Debug 
where 
    C: Clone + Debug + 'static,
{
    type Args: Clone + Debug + 'static;
    type RetVal: Clone + Debug + 'static;
    type RetDiff: Clone + Debug + 'static;
    type TraceType: Trace<C, Args = Self::Args, RetVal = Self::RetVal, RetDiff = Self::RetDiff> + 'static;

    /// Get the return value type information
    /// Equivalent to Julia's get_return_type(::GenerativeFunction{T,U}) where {T,U} = T
    fn get_return_type(&self) -> TypeId {
        TypeId::of::<Self::RetVal>()
    }

    /// Get the trace type information  
    /// Equivalent to Julia's get_trace_type(::GenerativeFunction{T,U}) where {T,U} = U
    fn get_trace_type(&self) -> TypeId {
        TypeId::of::<Self::TraceType>()
    }

    /// Get the return value type name as a string (for debugging/display)
    fn get_return_type_name(&self) -> &'static str {
        type_name::<Self::RetVal>()
    }

    /// Get the trace type name as a string (for debugging/display)
    fn get_trace_type_name(&self) -> &'static str {
        type_name::<Self::TraceType>()
    }

    /// Execute the generative function and return the trace.
    ///
    /// Given arguments, sample (r, t) ~ p(·; x) and return a trace with choice map t.
    fn simulate(&self, rng: Arc<Mutex<StdRng>>, args: Self::Args) -> Self::TraceType;

    /// Return a trace of a generative function that is consistent with the given constraints.
    ///
    /// Given arguments x and assignment u (constraints), sample t ~ q(·; u, x) and r ~ q(·; x, t),
    /// and return the trace (x, r, t) and the weight: log(p(r, t; x) / (q(t; u, x) * q(r; x, t)))
    fn generate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        constraints: ChoiceMap<C>,
    ) -> Result<(Self::TraceType, f64), GFIError>;


    /// Sample an assignment and compute the probability of proposing that assignment.
    ///
    /// Given arguments, sample t ~ p(·; x) and r ~ p(·; x, t),
    /// and return t (choices) and the weight: log(p(r, t; x) / q(r; x, t))
    fn propose(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
    ) -> Result<(ChoiceMap<C>, f64, Self::RetVal), GFIError> {
        let trace = self.simulate(rng, args);
        let weight = trace.get_score();
        let choices = trace.get_choices();
        let retval = trace.get_retval().clone();
        Ok((choices, weight, retval))
    }

    /// Return the probability of proposing an assignment.
    ///
    /// Given arguments x and an assignment t (choices) such that p(t; x) > 0,
    /// sample r ~ q(·; x, t) and return the weight: log(p(r, t; x) / q(r; x, t))
    fn assess(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        choices: ChoiceMap<C>,
    ) -> Result<(f64, Self::RetVal), GFIError> {
        let (trace, weight) = self.generate(rng, args, choices)?;
        Ok((weight, trace.get_retval().clone()))
    }
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::address::Address;
    use crate::core::address::ChoiceMap;

    // Example implementation for testing
    #[derive(Debug)]
    struct MockTrace {
        args: (i32, i32),
        retval: f64,
        choices: ChoiceMap<f64>,
        score: f64,
    }

    impl Trace<f64> for MockTrace {
        type Args = (i32, i32);
        type RetVal = f64;
        type RetDiff = RetDiff;

        fn get_args(&self) -> &Self::Args {
            &self.args
        }

        fn get_retval(&self) -> &Self::RetVal {
            &self.retval
        }

        fn get_choices(&self) -> ChoiceMap<f64> {
            self.choices.clone()
        }

        fn get_score(&self) -> f64 {
            self.score
        }

        fn get_gen_fn(
            &self,
        ) -> &dyn GenerativeFunction<
            f64,
            Args = Self::Args,
            RetVal = Self::RetVal,
            RetDiff = Self::RetDiff,
            TraceType = Self,
        > {
            unimplemented!()
        }

        fn project(&self, _selection: Selection) -> f64 {
            unimplemented!()
        }

        fn update(
            &self,
            _rng: Arc<Mutex<StdRng>>,
            _args: Self::Args,
            _argdiffs: Vec<ArgDiff>,
            _constraints: ChoiceMap<f64>,
        ) -> Result<(Self, f64, Self::RetDiff, ChoiceMap<f64>), GFIError> {
            todo!("MockTrace update")
        }

        fn regenerate(
            &self,
            _rng: Arc<Mutex<StdRng>>,
            _args: Self::Args,
            _argdiffs: Vec<ArgDiff>,
            _selection: Selection,
        ) -> Result<(Self, f64, Self::RetDiff), GFIError> {
            unimplemented!()
        }
    }

    #[test]
    fn test_trace_basic_operations() {
        let mut choices = ChoiceMap::Empty;
        // Add a choice using the proper method - cast to the expected Any type
        choices = choices.at(&[Address::from("z")]).set(50.0);

        let trace = MockTrace {
            args: (2, 4),
            retval: 3.14,
            choices,
            score: -1.5,
        };

        assert_eq!(trace.get_args(), &(2, 4));
        assert_eq!(trace.get_retval(), &3.14);
        assert_eq!(trace.get_score(), -1.5);
        
        // Test that we can retrieve the value and it's the right type
        let retrieved = trace.get_value(&Address::from("z"));
        assert!(retrieved.is_some());
        if let Some(val) = retrieved {
            assert_eq!(val, 50.0);
        }
        
        // Test that nonexistent addresses return None
        assert!(trace.get_value(&Address::from("nonexistent")).is_none());
    }


    // Mock GenerativeFunction for testing type methods
    #[derive(Debug)]
    struct MockGenerativeFunction;

    impl GenerativeFunction<f64> for MockGenerativeFunction {
        type Args = (i32, i32);
        type RetVal = f64;
        type RetDiff = RetDiff;
        type TraceType = MockTrace;

        fn simulate(&self, _rng: Arc<Mutex<StdRng>>, args: Self::Args) -> Self::TraceType {
            MockTrace {
                args,
                retval: 0.0,
                choices: ChoiceMap::Empty,
                score: 0.0,
            }
        }

        fn generate(
            &self,
            _rng: Arc<Mutex<StdRng>>,
            _args: Self::Args,
            _constraints: ChoiceMap<f64>,
        ) -> Result<(Self::TraceType, f64), GFIError> {
            todo!("MockGenerativeFunction generate")
        }
    }

    #[test]
    fn test_generative_function_type_methods() {
        let gen_fn = MockGenerativeFunction;

        // Test type IDs (Julia equivalent: get_return_type, get_trace_type)
        assert_eq!(gen_fn.get_return_type(), TypeId::of::<f64>());
        assert_eq!(gen_fn.get_trace_type(), TypeId::of::<MockTrace>());

        // Test type names (useful for debugging)
        assert_eq!(gen_fn.get_return_type_name(), "f64");
        assert!(gen_fn.get_trace_type_name().contains("MockTrace"));

        // Verify they're different types
        assert_ne!(gen_fn.get_return_type(), gen_fn.get_trace_type());
    }
}
