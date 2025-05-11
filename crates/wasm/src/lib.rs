use rand::Rng;
use rand::prelude::*;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use js_sys::Float64Array;
use std::f64::consts::PI;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

// For debugging
macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

#[derive(Serialize, Deserialize, Clone)]
struct Distribution {
    x: f64,
    pdf: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct HistogramBin {
    x: f64,
    frequency: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct SampleStep {
    x: f64,
    accepted: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct SimulationState {
    current_x: f64,
    proposed_x: f64,
    acceptance_ratio: f64,
    samples: Vec<f64>,
    steps: Vec<SampleStep>,
    distribution: Vec<Distribution>,
    histogram: Vec<HistogramBin>,
}

#[wasm_bindgen]
pub struct MetropolisHastings {
    // GMM parameters
    mean1: f64,
    mean2: f64,
    variance1: f64,
    variance2: f64,
    mixture_weight: f64,
    
    // Algorithm parameters
    proposal_std_dev: f64,
    rng: ThreadRng,
    
    // State
    current_x: f64,
    proposed_x: Option<f64>,
    burn_in: usize,
    num_samples: usize,
    current_step: usize,
    accepted_count: usize,
    samples: Vec<f64>,
    steps: Vec<SampleStep>,
}

#[wasm_bindgen]
impl MetropolisHastings {
    #[wasm_bindgen(constructor)]
    pub fn new(
        mean1: f64,
        mean2: f64,
        variance1: f64,
        variance2: f64,
        mixture_weight: f64,
        proposal_std_dev: f64,
        burn_in: usize,
        num_samples: usize
    ) -> Self {
        let mut rng = rand::thread_rng();
        
        // Start at the weighted mean
        let current_x = mixture_weight * mean1 + (1.0 - mixture_weight) * mean2;
        
        Self {
            mean1,
            mean2,
            variance1,
            variance2,
            mixture_weight,
            proposal_std_dev,
            rng,
            current_x,
            proposed_x: None,
            burn_in,
            num_samples,
            current_step: 0,
            accepted_count: 0,
            samples: Vec::new(),
            steps: Vec::new(),
        }
    }
    
    // Update parameters
    #[wasm_bindgen]
    pub fn update_parameters(
        &mut self,
        mean1: f64,
        mean2: f64,
        variance1: f64,
        variance2: f64,
        mixture_weight: f64,
        proposal_std_dev: f64,
        burn_in: usize,
        num_samples: usize
    ) {
        self.mean1 = mean1;
        self.mean2 = mean2;
        self.variance1 = variance1;
        self.variance2 = variance2;
        self.mixture_weight = mixture_weight;
        self.proposal_std_dev = proposal_std_dev;
        self.burn_in = burn_in;
        self.num_samples = num_samples;
        
        // Reset the state
        self.reset();
    }
    
    // Reset the simulation
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.current_x = self.mixture_weight * self.mean1 + (1.0 - self.mixture_weight) * self.mean2;
        self.proposed_x = None;
        self.current_step = 0;
        self.accepted_count = 0;
        self.samples.clear();
        self.steps.clear();
    }
    
    // Calculate the target PDF (GMM with two components)
    fn target_pdf(&self, x: f64) -> f64 {
        let gaussian1 = self.gaussian_pdf(x, self.mean1, self.variance1);
        let gaussian2 = self.gaussian_pdf(x, self.mean2, self.variance2);
        self.mixture_weight * gaussian1 + (1.0 - self.mixture_weight) * gaussian2
    }
    
    // Calculate a single Gaussian PDF
    fn gaussian_pdf(&self, x: f64, mean: f64, variance: f64) -> f64 {
        let exponent = -0.5 * (x - mean).powi(2) / variance;
        (1.0 / (2.0 * PI * variance).sqrt()) * exponent.exp()
    }
    
    // Run a single step of the algorithm
    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        // Propose a new sample from the proposal distribution (random walk)
        let proposed = self.current_x + self.rng.gen_range(-self.proposal_std_dev..self.proposal_std_dev);
        self.proposed_x = Some(proposed);
        
        // Calculate acceptance ratio
        let target_current = self.target_pdf(self.current_x);
        let target_proposed = self.target_pdf(proposed);
        
        // Symmetric proposal, so proposal densities cancel out
        let accept_prob = (target_proposed / target_current).min(1.0);
        
        // Accept or reject the proposal
        let u: f64 = self.rng.gen();
        let mut accepted = false;
        
        if u < accept_prob {
            self.current_x = proposed;
            accepted = true;
            self.accepted_count += 1;
        }
        
        // Record the step
        self.steps.push(SampleStep {
            x: self.current_x,
            accepted,
        });
        
        // Only add to samples after burn-in period
        if self.current_step >= self.burn_in {
            self.samples.push(self.current_x);
        }
        
        // Increment step counter
        self.current_step += 1;
        
        // Return whether we've reached the desired number of samples
        self.current_step < (self.num_samples + self.burn_in)
    }
    
    // Run multiple steps at once (for better performance)
    #[wasm_bindgen]
    pub fn run_steps(&mut self, num_steps: usize) -> bool {
        let mut more_steps = true;
        for _ in 0..num_steps {
            more_steps = self.step();
            if !more_steps {
                break;
            }
        }
        more_steps
    }
    
    // Get the current state
    #[wasm_bindgen]
    pub fn get_state_json(&self) -> String {
        let acceptance_ratio = if self.current_step > 0 {
            self.accepted_count as f64 / self.current_step as f64
        } else {
            0.0
        };
        
        let state = SimulationState {
            current_x: self.current_x,
            proposed_x: self.proposed_x.unwrap_or(self.current_x),
            acceptance_ratio,
            samples: self.samples.clone(),
            steps: self.steps.clone(),
            distribution: self.generate_distribution_data(),
            histogram: self.generate_histogram_data(),
        };
        
        serde_json::to_string(&state).unwrap_or_else(|_| "{}".to_string())
    }
    
    // Generate the target distribution data for plotting
    fn generate_distribution_data(&self) -> Vec<Distribution> {
        let min = self.mean1.min(self.mean2) - 4.0 * self.variance1.sqrt().max(self.variance2.sqrt());
        let max = self.mean1.max(self.mean2) + 4.0 * self.variance1.sqrt().max(self.variance2.sqrt());
        let step = (max - min) / 100.0;
        
        let mut points = Vec::new();
        let mut x = min;
        
        while x <= max {
            points.push(Distribution {
                x,
                pdf: self.target_pdf(x),
            });
            x += step;
        }
        
        points
    }
    
    // Generate histogram data from the samples
    fn generate_histogram_data(&self) -> Vec<HistogramBin> {
        if self.samples.is_empty() {
            return Vec::new();
        }
        
        // Find min and max values
        let min = self.samples.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 0.5;
        let max = self.samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 0.5;
        
        // Create bins
        let bin_count = (self.samples.len() as f64).sqrt().ceil().min(30.0) as usize;
        let bin_width = (max - min) / bin_count as f64;
        
        let mut bins = vec![0; bin_count];
        
        // Count samples in each bin
        for &sample in &self.samples {
            let bin_idx = ((sample - min) / bin_width).floor() as usize;
            if bin_idx < bin_count {
                bins[bin_idx] += 1;
            }
        }
        
        // Convert to frequency
        let total_samples = self.samples.len() as f64;
        let mut histogram = Vec::new();
        
        for (i, &count) in bins.iter().enumerate() {
            let x = min + (i as f64 + 0.5) * bin_width;
            let frequency = count as f64 / (total_samples * bin_width);
            
            histogram.push(HistogramBin { x, frequency });
        }
        
        histogram
    }
    
    // Get sample statistics
    #[wasm_bindgen]
    pub fn get_sample_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return f64::NAN;
        }
        
        let sum: f64 = self.samples.iter().sum();
        sum / self.samples.len() as f64
    }
    
    #[wasm_bindgen]
    pub fn get_sample_variance(&self) -> f64 {
        if self.samples.len() <= 1 {
            return f64::NAN;
        }
        
        let mean = self.get_sample_mean();
        let sum_sq_diff: f64 = self.samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
            
        sum_sq_diff / (self.samples.len() - 1) as f64
    }
    
    // Get arrays for JavaScript
    #[wasm_bindgen]
    pub fn get_samples(&self) -> Float64Array {
        let array = Float64Array::new_with_length(self.samples.len() as u32);
        for (i, &sample) in self.samples.iter().enumerate() {
            array.set_index(i as u32, sample);
        }
        array
    }
    
    #[wasm_bindgen]
    pub fn get_current_step(&self) -> usize {
        self.current_step
    }
    
    #[wasm_bindgen]
    pub fn get_samples_count(&self) -> usize {
        self.samples.len()
    }
    
    #[wasm_bindgen]
    pub fn get_acceptance_ratio(&self) -> f64 {
        if self.current_step > 0 {
            self.accepted_count as f64 / self.current_step as f64
        } else {
            0.0
        }
    }
    
    // Get component-specific means
    #[wasm_bindgen]
    pub fn get_component_means(&self) -> Float64Array {
        if self.samples.is_empty() {
            let array = Float64Array::new_with_length(2);
            array.set_index(0, f64::NAN);
            array.set_index(1, f64::NAN);
            return array;
        }
        
        // Calculate the midpoint between means
        let midpoint = (self.mean1 + self.mean2) / 2.0;
        
        // Split samples into two components
        let (comp1_samples, comp2_samples): (Vec<f64>, Vec<f64>) = self.samples
            .iter()
            .partition(|&&x| x < midpoint);
        
        // Calculate means for each component
        let comp1_mean = if comp1_samples.is_empty() {
            f64::NAN
        } else {
            comp1_samples.iter().sum::<f64>() / comp1_samples.len() as f64
        };
        
        let comp2_mean = if comp2_samples.is_empty() {
            f64::NAN
        } else {
            comp2_samples.iter().sum::<f64>() / comp2_samples.len() as f64
        };
        
        let array = Float64Array::new_with_length(2);
        array.set_index(0, comp1_mean);
        array.set_index(1, comp2_mean);
        array
    }
    
    // Get component-specific samples
    #[wasm_bindgen]
    pub fn get_component_samples(&self) -> js_sys::Array {
        if self.samples.is_empty() {
            let array = js_sys::Array::new();
            array.push(&Float64Array::new_with_length(0));
            array.push(&Float64Array::new_with_length(0));
            return array;
        }
        
        // Calculate the midpoint between means
        let midpoint = (self.mean1 + self.mean2) / 2.0;
        
        // Split samples into two components
        let (comp1_samples, comp2_samples): (Vec<f64>, Vec<f64>) = self.samples
            .iter()
            .cloned()
            .partition(|&x| x < midpoint);
        
        // Convert to Float64Array
        let comp1_array = Float64Array::new_with_length(comp1_samples.len() as u32);
        for (i, &sample) in comp1_samples.iter().enumerate() {
            comp1_array.set_index(i as u32, sample);
        }
        
        let comp2_array = Float64Array::new_with_length(comp2_samples.len() as u32);
        for (i, &sample) in comp2_samples.iter().enumerate() {
            comp2_array.set_index(i as u32, sample);
        }
        
        let array = js_sys::Array::new();
        array.push(&comp1_array);
        array.push(&comp2_array);
        array
    }
}