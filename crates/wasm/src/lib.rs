use rand::Rng;
use rand::prelude::*;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use js_sys::Float64Array;
use std::f64::consts::PI;
use ppl::{GenerativeFunction, Trace, Value, Expression, mh, parse_string};
use std::collections::{HashMap, HashSet};

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
struct Step {
    mu1: f64,
    mu2: f64,
    accepted: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct SimulationState {
    mu1: f64,
    mu2: f64,
    acceptance_ratio: f64,
    samples: Samples,
    steps: Vec<Step>,
    distribution: Vec<Distribution>,
    histogram: Vec<HistogramBin>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Samples {
    mu1: Vec<f64>,
    mu2: Vec<f64>,
}


#[wasm_bindgen]
pub struct Simulator {
    program: GenerativeFunction,
    trace: Option<Trace<GenerativeFunction, String, Value>>,
    scales: HashMap<String, f64>,
    selection: HashSet<String>,
    current_step: usize,
    accepted_count: usize,
    steps: Vec<Step>,
    burn_in: usize,
}

#[wasm_bindgen]
impl Simulator {
    #[wasm_bindgen(constructor)]
    pub fn new(model: &str) -> Result<Simulator, String> {
        // Parse the model string into PPL expressions
        let exprs = parse_string(model);
        
        // Create scales for parameters
        let mut scales = HashMap::new();
        scales.insert("mu1".to_string(), 1.0);
        scales.insert("mu2".to_string(), 1.0);
        
        // Create the program
        let program = GenerativeFunction::new(
            exprs,
            vec!["data".to_string()],
            scales.clone(),
            42
        );
        
        Ok(Simulator {
            program,
            trace: None,
            scales,
            selection: HashSet::new(),
            current_step: 0,
            accepted_count: 0,
            steps: Vec::new(),
            burn_in: 100, // Default burn-in period
        })
    }
    
    #[wasm_bindgen]
    pub fn initialize(&mut self, data: &[f64]) -> Result<(), String> {
        // Convert data to PPL Value
        let wrapped_data = Value::List(
            data.iter()
                .map(|&x| Value::Float(x))
                .collect()
        );
        
        // Simulate initial trace
        self.trace = Some(self.program.simulate(vec![wrapped_data])?);

        println!("Trace: {:?}", self.trace);
        
        // Set up selection for parameters
        self.selection = HashSet::from_iter(
            vec!["mu1".to_string(), "mu2".to_string()]
        );
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn step(&mut self) -> Result<bool, String> {
        if let Some(trace) = &self.trace {
            let mu1 = trace.get_choice(&"mu1".to_string()).value.expect_float();
            let mu2 = trace.get_choice(&"mu2".to_string()).value.expect_float();
            
            let (new_trace, accepted) = mh(
                self.program.clone(),
                trace.clone(),
                self.selection.clone()
            )?;
            
            // Record the step
            self.steps.push(Step {
                mu1,
                mu2,
                accepted,
            });
            
            if accepted {
                self.accepted_count += 1;
            }
            
            self.current_step += 1;
            self.trace = Some(new_trace);
            Ok(accepted)
        } else {
            Err("Simulator not initialized".to_string())
        }
    }
    
    #[wasm_bindgen]
    pub fn get_current_x(&self) -> Result<f64, String> {
        if let Some(trace) = &self.trace {
            let mu1 = trace.get_choice(&"mu1".to_string()).value.expect_float();
            let mu2 = trace.get_choice(&"mu2".to_string()).value.expect_float();
            Ok((mu1 + mu2) / 2.0) // Return average of parameters as current x
        } else {
            Err("Simulator not initialized".to_string())
        }
    }
    
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> Result<Float64Array, String> {
        if let Some(trace) = &self.trace {
            let mu1 = trace.get_choice(&"mu1".to_string()).value.expect_float();
            let mu2 = trace.get_choice(&"mu2".to_string()).value.expect_float();
            
            let result = Float64Array::new_with_length(2);
            result.set_index(0, mu1);
            result.set_index(1, mu2);
            Ok(result)
        } else {
            Err("Simulator not initialized".to_string())
        }
    }
    
    #[wasm_bindgen]
    pub fn get_state_json(&self) -> String {
        let acceptance_ratio = if self.current_step > 0 {
            self.accepted_count as f64 / self.current_step as f64
        } else {
            0.0
        };
        
        let (mu1, mu2) = if let Some(trace) = &self.trace {
            (
                trace.get_choice(&"mu1".to_string()).value.clone().expect_float(),
                trace.get_choice(&"mu2".to_string()).value.clone().expect_float()
            )
        } else {
            (0.0, 0.0)
        };
        
        let state = SimulationState {
            mu1,
            mu2,
            acceptance_ratio,
            samples: Samples {
                mu1: self.steps.iter().map(|s| s.mu1).collect(),
                mu2: self.steps.iter().map(|s| s.mu2).collect(),
            },
            steps: self.steps.clone(),
            distribution: self.generate_distribution_data(),
            histogram: self.generate_histogram_data(),
        };
        
        serde_json::to_string(&state).unwrap_or_else(|_| "{}".to_string())
    }
    
    // Reuse existing helper methods from MetropolisHastings
    fn generate_distribution_data(&self) -> Vec<Distribution> {
        let min = -4.0;
        let max = 4.0;
        let step = (max - min) / 100.0;
        
        let mut points = Vec::new();
        let mut x = min;
        
        while x <= max {
            let pdf = self.target_pdf(x);
            points.push(Distribution { x, pdf });
            x += step;
        }
        
        points
    }
    
    fn generate_histogram_data(&self) -> Vec<HistogramBin> {
        if self.steps.is_empty() {
            return Vec::new();
        }
        
        let min = self.steps.iter().fold(f64::INFINITY, |a, b| a.min(b.mu1).min(b.mu2)) - 0.5;
        let max = self.steps.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b.mu1).max(b.mu2)) + 0.5;
        let range = max - min;
        let num_bins = 20;
        let bin_width = range / num_bins as f64;
        
        let mut bins = vec![0.0; num_bins];
        let mut total = 0.0;
        
        for step in self.steps.iter() {
            let bin_index = ((step.mu1 - min) / bin_width).floor() as usize;
            if bin_index < num_bins {
                bins[bin_index] += 1.0;
                total += 1.0;
            }
            let bin_index = ((step.mu2 - min) / bin_width).floor() as usize;
            if bin_index < num_bins {
                bins[bin_index] += 1.0;
                total += 1.0;
            }
        }
        
        bins.iter().enumerate().map(|(i, &count)| {
            let x = min + (i as f64 + 0.5) * bin_width;
            let frequency = count / total;
            HistogramBin { x, frequency }
        }).collect()
    }
    
    fn target_pdf(&self, x: f64) -> f64 {
        if let Some(trace) = &self.trace {
            let mu1 = trace.get_choice(&"mu1".to_string()).value.expect_float();
            let mu2 = trace.get_choice(&"mu2".to_string()).value.expect_float();
            let gaussian1 = self.gaussian_pdf(x, mu1, 1.0);
            let gaussian2 = self.gaussian_pdf(x, mu2, 1.0);
            0.5 * gaussian1 + 0.5 * gaussian2
        } else {
            0.0
        }
    }
    
    fn gaussian_pdf(&self, x: f64, mean: f64, variance: f64) -> f64 {
        let exponent = -0.5 * (x - mean).powi(2) / variance;
        (1.0 / (2.0 * PI * variance).sqrt()) * exponent.exp()
    }
    
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accepted_count = 0;
        self.steps.clear();
        self.trace = None;
    }
    
    #[wasm_bindgen]
    pub fn get_sample_mean(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.iter().fold(0.0, |sum, step| sum + step.mu1) / self.steps.len() as f64
    }
    
    #[wasm_bindgen]
    pub fn get_sample_variance(&self) -> f64 {
        if self.steps.len() < 2 {
            return 0.0;
        }
        let mean = self.get_sample_mean();
        self.steps.iter()
            .map(|step| (step.mu1 - mean).powi(2))
            .sum::<f64>() / (self.steps.len() - 1) as f64
    }
    
    #[wasm_bindgen]
    pub fn get_samples(&self) -> Float64Array {
        let result = Float64Array::new_with_length(self.steps.len() as u32);
        for (i, step) in self.steps.iter().enumerate() {
            result.set_index(i as u32, step.mu1);
        }
        result
    }
    
    #[wasm_bindgen]
    pub fn get_current_step(&self) -> usize {
        self.current_step
    }
    
    #[wasm_bindgen]
    pub fn get_samples_count(&self) -> usize {
        self.steps.len()
    }
    
    #[wasm_bindgen]
    pub fn get_acceptance_ratio(&self) -> f64 {
        if self.current_step == 0 {
            return 0.0;
        }
        self.accepted_count as f64 / self.current_step as f64
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator() {
        let model = r#"
        (
            (sample mu1 (normal 0.0 1.0))
            (sample mu2 (normal 0.0 1.0))
            (constrain (< mu1 mu2))
            (define p 0.5)
            (define mix (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p))))
            (define observe-point (lambda (x) (observe (gensym) mix x)))
            (for-each observe-point data)
        )
        "#;
        let mut simulator = Simulator::new(model).unwrap();
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        simulator.initialize(&data).unwrap();
    }
}