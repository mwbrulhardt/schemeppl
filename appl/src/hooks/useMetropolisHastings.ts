import { useState, useEffect, useCallback } from 'react';
import init, { MetropolisHastings } from '@/pkg/wasm';

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
  current_x: number;
  proposed_x: number;
  acceptance_ratio: number;
  samples: number[];
  steps: Array<{ x: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
}

export function useMetropolisHastings() {
  const [algorithm, setAlgorithm] = useState<MetropolisHastings | null>(null);
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

  // Initialize WASM
  useEffect(() => {
    init().then(() => {
      const algo = new MetropolisHastings(
        parameters.mean1,
        parameters.mean2,
        parameters.variance1,
        parameters.variance2,
        parameters.mixtureWeight,
        parameters.proposalStdDev,
        parameters.burnIn,
        parameters.numSamples
      );
      setAlgorithm(algo);
      updateState(algo);
    }).catch(error => {
      console.error('Failed to initialize WASM:', error);
    });
  }, []);

  // Update state from algorithm
  const updateState = useCallback((algo: MetropolisHastings) => {
    try {
      const stateJson = algo.get_state_json();
      setState(JSON.parse(stateJson));
    } catch (error) {
      console.error('Error updating state:', error);
    }
  }, []);

  // Update parameters
  const updateParameters = useCallback((newParams: Partial<Parameters>) => {
    setParameters(prev => {
      const updated = { ...prev, ...newParams };
      if (algorithm) {
        algorithm.update_parameters(
          updated.mean1,
          updated.mean2,
          updated.variance1,
          updated.variance2,
          updated.mixtureWeight,
          updated.proposalStdDev,
          updated.burnIn,
          updated.numSamples
        );
        updateState(algorithm);
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
    if (!algorithm) {
      console.error('No algorithm available');
      return;
    }

    const elapsed = timestamp - lastTime;
    if (elapsed > parameters.delay) {
      setLastTime(timestamp);
      try {
        const moreSteps = algorithm.step();
        updateState(algorithm);
        if (!moreSteps) {
          console.log('Simulation complete');
          pause();
          return;
        }
      } catch (error) {
        console.error('Error in simulation step:', error);
        pause();
        return;
      }
    }

    const id = requestAnimationFrame(runSimulation);
    setAnimationId(id);
  }, [algorithm, lastTime, parameters.delay, pause, updateState]);

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
    algorithm.reset();
    updateState(algorithm);
  }, [algorithm, pause, updateState]);

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