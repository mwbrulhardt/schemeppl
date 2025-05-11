import { useState, useEffect, useCallback, useRef } from 'react';
import init, { Simulator } from '@/pkg/wasm';

interface Parameters {
  mean1: number;
  mean2: number;
  variance1: number;
  variance2: number;
  mixtureWeight: number;
  proposalStdDev: number;
  numSamples: number;
  burnIn: number;
  delay: number;
}

interface SimulationState {
  mu1: number;
  mu2: number;
  acceptance_ratio: number;
  samples: {
    mu1: number[];
    mu2: number[];
  };
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
}

// Helper for normal sampling (Box-Muller transform)
function randn_bm() {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function createModelAndData(params: Parameters) {
  const data = Float64Array.from(
    Array.from({ length: 100 }, () => {
      const r = Math.random();
      return r < params.mixtureWeight
        ? params.mean1 + randn_bm() * Math.sqrt(params.variance1)
        : params.mean2 + randn_bm() * Math.sqrt(params.variance2);
    })
  );
  const model = `(
    (sample mu1 (normal 0.0 1.0))
    (sample mu2 (normal 0.0 1.0))
    (constrain (< mu1 mu2))
    (define p ${params.mixtureWeight.toFixed(6)})
    (define mix (mixture (list (normal mu1 ${Math.sqrt(params.variance1).toFixed(6)}) (normal mu2 ${Math.sqrt(params.variance2).toFixed(6)})) (list p (- 1.0 p))))
    (define observe-point (lambda (x) (observe (gensym) mix x)))
    (for-each observe-point data)
  )`;
  console.log('Generated model:', model);
  return { model, data };
}

export function useSimulator() {
  const [algorithm, setAlgorithm] = useState<Simulator | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [animationId, setAnimationId] = useState<number | null>(null);
  const [lastTime, setLastTime] = useState(0);
  const [state, setState] = useState<SimulationState | null>(null);
  const [parameters, setParameters] = useState<Parameters>({
    mean1: -2,
    mean2: 2,
    variance1: 1,
    variance2: 1,
    mixtureWeight: 0.5,
    proposalStdDev: 1,
    numSamples: 1000,
    burnIn: 100,
    delay: 10
  });

  // Ref to always have the latest isRunning value
  const isRunningRef = useRef(isRunning);
  useEffect(() => { isRunningRef.current = isRunning; }, [isRunning]);

  // Initialize WASM
  useEffect(() => {
    init().then(() => {
      // Use the latest parameters for initial model/data
      const { model, data } = createModelAndData(parameters);
      const algo = new Simulator(model);
      algo.initialize(data);
      setAlgorithm(algo);
      updateState(algo);
    }).catch(error => {
      console.error('Failed to initialize WASM:', error);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update state from algorithm
  const updateState = useCallback((algo: Simulator) => {
    try {
      const stateJson = algo.get_state_json();
      setState(JSON.parse(stateJson));
    } catch (error) {
      console.error('Error updating state:', error);
    }
  }, []);

  // Update parameters and reinitialize simulator
  const updateParameters = useCallback((newParams: Partial<Parameters>) => {
    setParameters(prev => {
      const updated = { ...prev, ...newParams };
      if (algorithm) {
        // Always generate new data and reinitialize the sampler
        const { model, data } = createModelAndData(updated);
        const algo = new Simulator(model);
        algo.initialize(data);
        setAlgorithm(algo);
        updateState(algo);
      }
      return updated;
    });
  }, [algorithm, updateState]);

  // Pause simulation
  const pause = useCallback(() => {
    setIsRunning(false);
    if (animationId) {
      cancelAnimationFrame(animationId);
      setAnimationId(null);
    }
  }, [animationId, setIsRunning]);

  // Run simulation loop
  const runSimulation = useCallback((timestamp = 0) => {
    if (!algorithm || !isRunningRef.current) {
      return;
    }

    const elapsed = timestamp - lastTime;
    if (elapsed > parameters.delay) {
      setLastTime(timestamp);
      try {
        algorithm.step();
        const newState = JSON.parse(algorithm.get_state_json());
        setState(newState);
        
        if (
          newState.samples.mu1.length < parameters.numSamples &&
          isRunningRef.current
        ) {
          const id = requestAnimationFrame(runSimulation);
          setAnimationId(id);
        } else {
          console.log('Simulation complete - reached sample limit or paused');
          pause();
        }
      } catch (error) {
        console.error('Error in simulation step:', error);
        pause();
      }
    } else if (isRunningRef.current) {
      const id = requestAnimationFrame(runSimulation);
      setAnimationId(id);
    }
  }, [algorithm, lastTime, parameters.delay, parameters.numSamples, pause]);

  // Start simulation
  const start = useCallback(() => {
    if (!algorithm) {
      console.error('Algorithm not initialized');
      return;
    }
    const now = performance.now();
    setLastTime(now);
    setIsRunning(true);
    setTimeout(() => {
      runSimulation(now);
    }, 0);
  }, [algorithm, runSimulation, setIsRunning]);

  // Reset simulation
  const reset = useCallback(() => {
    if (!algorithm) return;
    pause();
    
    // Generate new data points
    const data = Float64Array.from(
      Array.from({ length: 100 }, () => {
        const r = Math.random();
        return r < 0.5 ? 
          parameters.mean1 + Math.random() * Math.sqrt(parameters.variance1) :
          parameters.mean2 + Math.random() * Math.sqrt(parameters.variance2);
      })
    );

    algorithm.reset();
    algorithm.initialize(data);
    updateState(algorithm);
  }, [algorithm, pause, parameters, updateState]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [animationId]);

  return {
    algorithm,
    isRunning,
    start,
    pause,
    reset,
    updateParameters,
    state,
    parameters
  };
} 