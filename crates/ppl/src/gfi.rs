use crate::address::{Address, Selection};
use ndarray::ArrayD;
use rand::rngs::StdRng;
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::Index;
use std::sync::Arc;
use std::sync::Mutex;

pub type Density = ArrayD<f64>;
pub type Weight = ArrayD<f64>;
pub type Score = ArrayD<f64>;

/// Abstract trait for a trace of a generative function.
///
/// A trace captures the execution of a generative function, including its arguments,
/// return value, choice map, and probability score.
///
/// # Type Parameters
///
/// * `X` - The choice map type that stores random choices made during execution
/// * `R` - The return value type
pub trait Trace<X, R>: Debug
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
{
    type Args;

    /// Returns the arguments used for this execution.
    fn get_args(&self) -> &Self::Args;

    /// Returns the return value of this execution.
    fn get_retval(&self) -> R;

    /// Returns the choice map for this trace.
    ///
    /// Returns owned values to enable on-demand construction and caching.
    fn get_choices(&self) -> X;

    /// Returns the log probability score: log(p(r, t; x) / q(r; x, t)).
    ///
    /// When there is no non-addressed randomness, this simplifies to log p(t; x).
    fn get_score(&self) -> f64;

    /// Returns the generative function that produced this trace.
    fn get_gen_fn(&self) -> &dyn GenerativeFunction<X, R, Args = Self::Args, TraceType = Self>;

    /// Updates this trace with new arguments and choice constraints.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `x` - New choice constraints
    /// * `args` - New arguments
    ///
    /// # Returns
    ///
    /// A tuple of (new_trace, weight, discarded_choices) where:
    /// - `new_trace` - Updated trace
    /// - `weight` - Importance weight for the update
    /// - `discarded_choices` - Old choice values that were changed
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if the update fails due to invalid choices or addresses.
    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        x: X,
        args: Self::Args,
    ) -> Result<(Self, Weight, X), GFIError>
    where
        Self: Sized;

    /// Gets a choice value by address.
    ///
    /// # Arguments
    ///
    /// * `addr` - The address to look up
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let value = trace.get(Address::from("x"));
    /// ```
    fn get(&self, addr: Address) -> <X as Index<Address>>::Output
    where
        <X as Index<Address>>::Output: Clone,
    {
        let choices = self.get_choices();
        choices[addr].clone()
    }
}

/// Errors that can occur during generative function interface operations.
#[derive(Debug, Clone)]
pub enum GFIError {
    /// Operation is not implemented.
    NotImplemented(String),
    /// Invalid choice value provided.
    InvalidChoice(String),
    /// Invalid address provided.
    InvalidAddress(String),
    /// Probability is zero (invalid state).
    ZeroProbability,
}

/// Abstract trait for a generative function.
///
/// A generative function defines a probability distribution over execution traces.
/// It can simulate new traces, generate traces with constraints, and update existing traces.
///
/// # Type Parameters
///
/// * `X` - The choice map type that stores random choices
/// * `R` - The return value type
pub trait GenerativeFunction<X, R>: Debug
where
    X: Clone + Debug + Index<Address> + 'static,
    R: Clone + Debug + 'static,
{
    type Args;
    type TraceType: Trace<X, R> + 'static;

    /// Returns the return value type information.
    fn get_return_type(&self) -> TypeId {
        TypeId::of::<R>()
    }

    /// Returns the trace type information.
    fn get_trace_type(&self) -> TypeId {
        TypeId::of::<Self::TraceType>()
    }

    /// Executes the generative function and returns a trace.
    ///
    /// Samples (r, t) ~ p(·; x) and returns a trace with choice map t.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `args` - Arguments to the generative function
    fn simulate(&self, rng: Arc<Mutex<StdRng>>, args: Self::Args) -> Self::TraceType;

    /// Generates a trace with optional constraints on some choices.
    ///
    /// Samples unconstrained choices and computes importance weight for inference.
    /// When constraints are None, equivalent to simulate() but returns weight=0.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `constraints` - Optional constraints on subset of choices
    /// * `args` - Arguments to the generative function
    ///
    /// # Returns
    ///
    /// A tuple of (trace, weight) where:
    /// - `trace` - Contains all choices (constrained + sampled) and return value
    /// - `weight` - Importance weight: log [P(all_choices; args) / Q(unconstrained_choices; constrained_choices, args)]
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if generation fails due to invalid constraints.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Generate with constraints
    /// let constraints = Some(choice_map);
    /// let (trace, weight) = model.generate(rng, constraints, args)?;
    /// ```
    fn generate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        constraints: Option<X>,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight), GFIError>;

    /// Computes the log probability density of given choices.
    ///
    /// Computes log P(choices; args) where P is the generative function's measure kernel.
    /// Also computes the return value for the given choices.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `args` - Arguments to the generative function
    /// * `choices` - The choices to evaluate
    ///
    /// # Returns
    ///
    /// A tuple of (log_density, retval) where:
    /// - `log_density` - log P(choices; args)
    /// - `retval` - Return value computed with the given choices
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if P(choices; args) = 0 (invalid choices).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let (log_density, retval) = model.assess(rng, args, choices)?;
    /// ```
    fn assess(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
        choices: X,
    ) -> Result<(Density, R), GFIError> {
        let (trace, weight) = self.generate(rng, Some(choices), args)?;
        Ok((weight, trace.get_retval().clone()))
    }

    /// Updates a trace with new arguments and/or choice constraints.
    ///
    /// Transforms trace from (old_args, old_choices) to (new_args, new_choices)
    /// and computes incremental importance weight for MCMC and SMC algorithms.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `trace` - Current trace to update
    /// * `constraints` - Optional constraints on choices to enforce during update
    /// * `args` - New arguments to the generative function
    ///
    /// # Returns
    ///
    /// A tuple of (new_trace, weight, discarded_choices) where:
    /// - `new_trace` - Updated trace with new arguments and choices
    /// - `weight` - Incremental importance weight for the update
    /// - `discarded_choices` - Old choice values that were changed
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if the update fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let (new_trace, weight, discarded) = model.update(rng, old_trace, None, new_args)?;
    /// ```
    fn update(
        &self,
        rng: Arc<Mutex<StdRng>>,
        trace: Self::TraceType,
        constraints: Option<X>,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight, X), GFIError>
    where
        Self: Sized;

    /// Regenerates selected choices in a trace while keeping others fixed.
    ///
    /// Resamples choices at addresses selected by 'selection' from their conditional distribution
    /// and computes incremental importance weight.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `trace` - Current trace to regenerate from
    /// * `selection` - Selection specifying which addresses to regenerate
    /// * `args` - Arguments to the generative function
    ///
    /// # Returns
    ///
    /// A tuple of (new_trace, weight, discarded_choices) where:
    /// - `new_trace` - Trace with selected choices resampled
    /// - `weight` - Incremental importance weight for the regeneration
    /// - `discarded_choices` - Old values of the regenerated choices
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if regeneration fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let selection = Selection::from_addresses(&["x", "y"]);
    /// let (new_trace, weight, discarded) = model.regenerate(rng, trace, selection, args)?;
    /// ```
    fn regenerate(
        &self,
        rng: Arc<Mutex<StdRng>>,
        trace: Self::TraceType,
        selection: Selection,
        args: Self::Args,
    ) -> Result<(Self::TraceType, Weight, Option<X>), GFIError>
    where
        Self: Sized;

    /// Sample an assignment and compute the probability of proposing that assignment.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `args` - Arguments to the generative function
    ///
    /// # Returns
    ///
    /// A tuple of (choices, density, retval) where:
    /// - `choices` - The sampled choice map
    /// - `density` - Log probability density: log P(choices; args)  
    /// - `retval` - Return value computed with the sampled choices
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if the proposal fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let (choices, log_density, retval) = proposal.propose(rng, args)?;
    /// ```
    fn propose(
        &self,
        rng: Arc<Mutex<StdRng>>,
        args: Self::Args,
    ) -> Result<(X, Density, R), GFIError> {
        let trace = self.simulate(rng, args);
        let weight = ndarray::Array::from_elem(ndarray::IxDyn(&[]), trace.get_score());
        Ok((trace.get_choices(), weight, trace.get_retval()))
    }

    /// Merges two choice maps.
    ///
    /// Used internally for compositional generative functions where choice maps
    /// from different components need to be combined. The merge operation resolves
    /// conflicts by preferring choices from `x_` over `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - First choice map
    /// * `x_` - Second choice map (takes precedence in conflicts)
    /// * `check` - Optional boolean array for conditional selection
    ///
    /// # Returns
    ///
    /// A tuple of (merged choice map, discarded values) where:
    /// - `merged` - Combined choices with `x_` values overriding `x` values at conflicts
    /// - `discarded` - Values from `x` that were overridden by `x_` (None if no conflicts)
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if the merge operation fails.
    fn merge(&self, x: X, x_: X, check: Option<ArrayD<f64>>) -> Result<(X, Option<X>), GFIError>
    where
        X: Clone + Debug + Index<Address> + 'static,
        Self: Sized;

    /// Filters choice map into selected and unselected parts.
    ///
    /// Partitions choices based on a selection, enabling fine-grained manipulation
    /// of subsets of choices in inference algorithms.
    ///
    /// # Arguments
    ///
    /// * `x` - Choice map to filter
    /// * `selection` - Selection specifying which addresses to include
    ///
    /// # Returns
    ///
    /// A tuple of (selected_choices, unselected_choices) where:
    /// - `selected_choices` - Choice map containing only selected addresses (None if no matches)
    /// - `unselected_choices` - Choice map containing only unselected addresses (None if no matches)
    ///
    /// # Errors
    ///
    /// Returns `GFIError` if the filter operation fails.
    fn filter(&self, x: X, selection: Selection) -> Result<(Option<X>, Option<X>), GFIError>
    where
        X: Clone + Debug + Index<Address> + 'static,
        Self: Sized;
}
