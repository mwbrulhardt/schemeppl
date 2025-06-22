use rand::rngs::StdRng;
use std::any::TypeId;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;

use crate::address::{Address, Selection};
use crate::choice_map::{ChoiceMap, ChoiceMapQuery};

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
pub trait Trace<R, C>: Debug
where
    R: Clone + Debug + 'static,
    C: Clone + Debug + 'static,
{
    type Args;
    type RetDiff;
    type ChoiceMap: ChoiceMap<C>;

    /// Return the argument tuple for a given execution.
    fn get_args(&self) -> &Self::Args;

    /// Return the return value of the given execution.
    fn get_retval(&self) -> &R;

    /// Get the choice map for this trace
    /// Returns owned values to enable on-demand construction and caching
    fn get_choices(&self) -> Self::ChoiceMap;

    /// Return log(p(r, t; x) / q(r; x, t))
    ///
    /// When there is no non-addressed randomness, this simplifies to the log probability log p(t; x).
    fn get_score(&self) -> f64;

    /// Return the generative function that produced the given trace.
    fn get_gen_fn(&self) -> &dyn GenerativeFunction<R, C, Args = Self::Args, TraceType = Self>;

    /// Get the value of the random choice at address `addr`.
    fn get_value(&self, addr: &Address) -> Option<C> {
        self.get_choices().get_value(addr).cloned()
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
        constraints: Self::ChoiceMap,
    ) -> Result<(Self, f64, Self::RetDiff, Self::ChoiceMap), GFIError>
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
/// - R is the return value type (RetVal)
/// - C is the choice value type (ChoiceValue)
pub trait GenerativeFunction<R, C>: Debug
where
    R: Clone + Debug + 'static,
    C: Clone + Debug + 'static,
{
    type Args;
    type TraceType: Trace<R, C> + 'static;

    /// Get the return value type information
    /// Equivalent to Julia's get_return_type(::GenerativeFunction{T,U}) where {T,U} = T
    fn get_return_type(&self) -> TypeId {
        TypeId::of::<R>()
    }

    /// Get the trace type information  
    /// Equivalent to Julia's get_trace_type(::GenerativeFunction{T,U}) where {T,U} = U
    fn get_trace_type(&self) -> TypeId {
        TypeId::of::<Self::TraceType>()
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
        constraints: <Self::TraceType as Trace<R, C>>::ChoiceMap,
    ) -> Result<(Self::TraceType, f64), GFIError>;

    /// Sample an assignment and compute the probability of proposing that assignment.
    ///
    /// Given arguments, sample t ~ p(·; x) and r ~ p(·; x, t),
    /// and return t (choices) and the weight: log(p(r, t; x) / q(r; x, t))
    fn propose(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
    ) -> Result<(<Self::TraceType as Trace<R, C>>::ChoiceMap, f64, R), GFIError> {
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
        choices: <Self::TraceType as Trace<R, C>>::ChoiceMap, // Use the trace's choice map type
    ) -> Result<(f64, R), GFIError> {
        let (trace, weight) = self.generate(rng, args, choices)?;
        Ok((weight, trace.get_retval().clone()))
    }
}
