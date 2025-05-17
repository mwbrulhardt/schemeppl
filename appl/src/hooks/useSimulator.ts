/* useSimulator.ts
 * React hook that animates a Metropolis-Hastings chain whose primitives live
 * in Rust/Wasm.  All Rust objects are *borrowed* across the boundary, so no
 * Rust values are dropped between frames.
 *
 * – `pause`     (leaf)            ──> no deps
 * – `runSimulation`               ──> depends on pause
 * – `start`                       ──> depends on runSimulation
 * – `reset` / `updateParameters`  ──> depend on pause (+ runSimulation for auto-restart)
 */

import init, {
  generate_data,
  JsGenerativeFunction,
  JsTrace,
  metropolis_hastings,
} from "@/pkg/wasm";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";


export interface Parameters {
  mu1: number;  mu2: number;
  sigma1: number; sigma2: number;
  p: number;
  proposalStdDev1: number;
  proposalStdDev2: number;
  numSteps: number;
  burnIn: number;
  delay: number;
  seed: number;
  sampleSize: number;
}

export interface SimulationState {
  mu1: number; mu2: number;
  acceptance_ratio: number;
  samples: { mu1: number[]; mu2: number[] };
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram:   Array<{ x: number; frequency: number }>;
  data: number[];
  labels: number[];
}


/*Create model for the given parameters*/
function createModel(p: Parameters) {
  return `(
    (sample mu1 (normal 0.0 1.0))
    (sample mu2 (normal 0.0 1.0))

    (constrain (< mu1 mu2))

    (define p ${p.p.toFixed(6)})
    (define mix (mixture (list (normal mu1 ${p.sigma1.toFixed(6)}) (normal mu2 ${p.sigma2.toFixed(6)})) (list p (- 1.0 p))))

    (define observe-point (lambda (x) (observe (gensym) mix x)))

    (for-each observe-point data)
  )`;
}


/*Create dataset for the given parameters*/
function createDataset(p: Parameters) {
  return generate_data(
    p.mu1, p.sigma1, p.mu2, p.sigma2,
    p.p, p.sampleSize, BigInt(p.seed)
  );
}

/* ─────────────────────────────────────────────────────────────── main hook */

export function useSimulator() {
  /* --------------- user-tunable params ---------------------------------- */
  const [parameters, setParameters] = useState<Parameters>({
    mu1: -2, mu2: 2,
    sigma1: 1, sigma2: 1,
    p: 0.5,
    proposalStdDev1: 0.1,
    proposalStdDev2: 0.1,
    numSteps: 1_000,
    burnIn: 100,
    delay: 0,
    seed: 42,
    sampleSize: 200,
  });

  /* --------------- wasm objects / chain state --------------------------- */
  const [algorithm,   setAlgorithm]   = useState<JsGenerativeFunction | null>(null);
  const [currentTrace, setCurrentTrace] = useState<JsTrace | null>(null);
  const traceRef = useRef<JsTrace | null>(null);    // always points at latest trace
  // Cache the current dataset to ensure consistency
  const datasetRef = useRef<{ data: Float64Array, labels: Uint8Array } | null>(null);
  // Keep a ref to the latest parameters to use in refreshDataset
  const paramsRef = useRef<Parameters>(parameters);

  // Update paramsRef whenever parameters change
  useEffect(() => {
    paramsRef.current = parameters;
  }, [parameters]);

  /* derived state for charts */
  const [simState, setSimState] = useState<SimulationState | null>(null);

  // Ref to gate UI updates (throttle)
  const lastUiUpdateRef = useRef<number>(0);

  /* --------------- animation bookkeeping -------------------------------- */
  const animationId      = useRef<number | null>(null);
  const isRunningRef     = useRef(false);
  const [isRunning, setIsRunning] = useState(false);

  const delayRef         = useRef(parameters.delay);
  const lastStepTimeRef  = useRef(0);
  const stepsRef         = useRef<Array<{ mu1: number; mu2: number; accepted: boolean }>>([]);
  const acceptedCountRef = useRef(0);
  const currentStepRef   = useRef(0);

  useEffect(() => { delayRef.current   = parameters.delay;   }, [parameters.delay]);
  useEffect(() => { isRunningRef.current = isRunning;        }, [isRunning]);
  useEffect(() => { traceRef.current     = currentTrace;     }, [currentTrace]);

  /* --------------- initial Wasm + first trace --------------------------- */
  useEffect(() => {
    (async () => {
      await init();

      // Create and cache the initial dataset with the current parameters
      const dataset = refreshDataset(parameters);
      const model = createModel(parameters);
      
      const gf = new JsGenerativeFunction(
        model,
        { mu1: parameters.proposalStdDev1, mu2: parameters.proposalStdDev2 },
        BigInt(parameters.seed)
      );

      /* finite-score first trace */
      const args = new Float64Array(dataset.data);
      let trace  = gf.simulate(args);
      for (let i = 0; !Number.isFinite(trace.score()) && i < 100; i++) {
        trace = gf.simulate(args);
      }

      setAlgorithm(gf);
      setCurrentTrace(trace);
      traceRef.current = trace;
      updateState(trace, [], 0, true);
    })().catch(console.error);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* --------------- helpers ---------------------------------------------- */
  const selection = useMemo(() => ["mu1", "mu2"] as const, []);

  // Create and cache a new dataset based on current parameters
  const refreshDataset = useCallback((params?: Parameters) => {
    // Use provided params or the latest ones from the ref
    const p = params || paramsRef.current;
    console.log('Refreshing dataset with params:', {
      sampleSize: p.sampleSize,
      mu1: p.mu1,
      mu2: p.mu2,
      sigma1: p.sigma1,
      sigma2: p.sigma2,
      p: p.p,
      seed: p.seed
    });

    const dataset = createDataset(p);
    // Debug the dataset structure
    console.log('Generated dataset:', {
      hasData: !!dataset.data,
      dataLength: dataset.data ? dataset.data.length : 0,
      hasLabels: !!dataset.labels,
      labelsLength: dataset.labels ? new Uint8Array(dataset.labels).length : 0,
      firstFewDataPoints: dataset.data ? Array.from(dataset.data).slice(0, 5) : []
    });

    // Store both data and labels in a consistent format
    datasetRef.current = {
      data: dataset.data,
      labels: new Uint8Array(dataset.labels)
    };
    
    if (datasetRef.current.data.length !== p.sampleSize) {
      console.warn(`Dataset size mismatch: expected ${p.sampleSize}, got ${datasetRef.current.data.length}`);
    }
    
    return datasetRef.current;
  }, []);

  const updateState = useCallback(
    (
      trace: JsTrace,
      steps: Array<{ mu1: number; mu2: number; accepted: boolean }>,
      accepted: number,
      force = false,
    ) => {
      // Throttle to ~15 fps (≈66 ms)
      const now = Date.now();
      if (!force && now - lastUiUpdateRef.current < 66) {
        return; // skip update
      }
      lastUiUpdateRef.current = now;

      const burn = Math.min(parameters.burnIn, steps.length);
      const post = steps.slice(burn);

      // Use the cached dataset - it should always be available at this point
      // If not, create it with current parameters
      const dataset = datasetRef.current || refreshDataset(parameters);

      setSimState({
        mu1: trace.get_choice("mu1"),
        mu2: trace.get_choice("mu2"),
        acceptance_ratio: steps.length === 0 ? 0 : accepted / steps.length,
        steps,
        samples: {
          mu1: post.map((s) => s.mu1),
          mu2: post.map((s) => s.mu2),
        },
        distribution: [],
        histogram: [],
        data: Array.from(dataset.data),
        labels: Array.from(dataset.labels),
      });
    },
    [parameters, refreshDataset]
  );

  /* ───────────────────────────────────────────────────────────── controls */

  /* ---------- pause (leaf) --------------------------------------------- */
  const pause = useCallback(() => {
    setIsRunning(false);
    if (animationId.current) cancelAnimationFrame(animationId.current);
    animationId.current = null;
  }, []);

  /* ---------- runSimulation (depends on pause) ------------------------- */
  const runSimulation = useCallback(function tick(ts = 0) {
    if (!algorithm || !traceRef.current || !isRunningRef.current) return;

    if (ts - lastStepTimeRef.current < delayRef.current) {
      animationId.current = requestAnimationFrame(tick);
      return;
    }

    try {
      const [nextTrace, accepted] = metropolis_hastings(
        algorithm,
        traceRef.current,
        [...selection],
      );
      traceRef.current = nextTrace;
      if (accepted) acceptedCountRef.current++;

      stepsRef.current.push({
        mu1: nextTrace.get_choice("mu1"),
        mu2: nextTrace.get_choice("mu2"),
        accepted,
      });
      currentStepRef.current++;

      /* cap memory */
      const cap = parameters.numSteps + parameters.burnIn + 10;
      if (stepsRef.current.length > cap) stepsRef.current.shift();

      updateState(nextTrace, stepsRef.current, acceptedCountRef.current);

      if (currentStepRef.current >= parameters.numSteps + parameters.burnIn) {
        pause();
      } else {
        lastStepTimeRef.current = ts;
        animationId.current = requestAnimationFrame(tick);
      }
    } catch (err) {
      console.error(err);
      pause();
    }
  },
  [algorithm, selection, parameters.numSteps, parameters.burnIn, updateState, pause]);

  /* ---------- start (depends on runSimulation) ------------------------- */
  const start = useCallback(() => {
    if (!algorithm || !traceRef.current) {
      console.error("Algorithm or trace not ready");
      return;
    }
    if (isRunningRef.current) return;

    delayRef.current        = parameters.delay;
    lastStepTimeRef.current = performance.now();
    setIsRunning(true);
    animationId.current = requestAnimationFrame(runSimulation);
  }, [algorithm, parameters.delay, runSimulation]);

  /* ---------- reset (depends on pause) --------------------------------- */
  const reset = useCallback(() => {
    pause();
    stepsRef.current = [];
    acceptedCountRef.current = 0;
    currentStepRef.current   = 0;

    // Refresh dataset with current parameters
    const dataset = refreshDataset(parameters);
    const model = createModel(parameters);
    
    const gf = new JsGenerativeFunction(
      model,
      { mu1: parameters.proposalStdDev1, mu2: parameters.proposalStdDev2 },
      BigInt(parameters.seed)
    );

    const args = new Float64Array(dataset.data);
    let trace  = gf.simulate(args);
    for (let i = 0; !Number.isFinite(trace.score()) && i < 100; i++) {
      trace = gf.simulate(args);
    }

    setAlgorithm(gf);
    setCurrentTrace(trace);
    traceRef.current = trace;
    updateState(trace, [], 0, true);
  }, [pause, parameters, updateState, refreshDataset]);

  /* ---------- updateParameters (depends on pause + runSimulation) ------ */
  const updateParameters = useCallback((delta: Partial<Parameters>) => {
    // Temporary debugging for sample size changes
    if ('sampleSize' in delta) {
      console.log('Sample size changing:', {
        from: parameters.sampleSize,
        to: delta.sampleSize,
        delta
      });
    }
    
    setParameters(prev => {
      const next = { ...prev, ...delta };

      // Validate all required parameters are valid numbers
      const requiredParams = ['mu1', 'mu2', 'sigma1', 'sigma2', 'p', 'proposalStdDev1', 'proposalStdDev2', 'numSteps', 'burnIn', 'delay', 'seed', 'sampleSize'];
      const isValid = requiredParams.every(param => 
        typeof next[param as keyof Parameters] === 'number' && 
        !isNaN(next[param as keyof Parameters])
      );

      if (!isValid) {
        return prev; // Keep previous valid state if new state is invalid
      }

      const structural =
        "mu1" in delta || "mu2" in delta ||
        "sigma1" in delta || "sigma2" in delta ||
        "p" in delta || "seed" in delta ||
        "proposalStdDev1" in delta || "proposalStdDev2" in delta ||
        "sampleSize" in delta;

      const wasRunning = isRunningRef.current;

      if (structural) pause();

      if (structural) {
        // Use the new parameters for model and dataset creation
        const model = createModel(next);
        
        // Explicitly refresh the dataset with the new parameters
        const dataset = refreshDataset(next);
        console.log('Refreshed dataset:', {
          params: next,
          dataLength: dataset.data.length,
          labelsLength: dataset.labels.length
        });
        
        const gf = new JsGenerativeFunction(
          model,
          { mu1: next.proposalStdDev1, mu2: next.proposalStdDev2 },
          BigInt(next.seed)
        );

        const args = new Float64Array(dataset.data);
        let trace  = gf.simulate(args);
        for (let i = 0; !Number.isFinite(trace.score()) && i < 100; i++) {
          trace = gf.simulate(args);
        }

        stepsRef.current         = [];
        acceptedCountRef.current = 0;
        currentStepRef.current   = 0;

        setAlgorithm(gf);
        setCurrentTrace(trace);
        traceRef.current = trace;
        updateState(trace, [], 0, true);

        if (wasRunning) {
          delayRef.current        = next.delay;
          lastStepTimeRef.current = performance.now();
          setIsRunning(true);
          animationId.current = requestAnimationFrame(runSimulation);
        }
      } else {
        /* non-structural change (just delay, numSteps, etc.) */
        delayRef.current = next.delay;
      }
      return next;
    });
  }, [pause, runSimulation, updateState, refreshDataset, parameters]);

  /* --------------- cleanup on unmount ---------------------------------- */
  useEffect(() => () => {
    if (animationId.current) cancelAnimationFrame(animationId.current);
  }, []);

  /* --------------- exposed API ------------------------------------------ */
  return {
    state: simState,
    parameters,
    start,
    pause,
    reset,
    updateParameters,
    isRunning,
    algorithm,
  };
}
