use std::fmt::Debug;

use statrs::distribution::Continuous;
use statrs::distribution::Discrete;

use rand::distributions::{Distribution, WeightedIndex};
use rand::RngCore;

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
        Self { flag }
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
    pub _marker: std::marker::PhantomData<T>,
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

#[cfg(test)]

mod tests {
    use super::*;
    use statrs::distribution::Normal;

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
}
