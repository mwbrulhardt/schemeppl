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
} from "@/pkg/wasm";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/* ───────────────────────────────────────────────────────────────── helpers */

function metropolisHastings(
  gf: JsGenerativeFunction,
  trace: JsTrace,
  selection: readonly string[],
): [JsTrace, boolean] {
  const [proposal, logW] = gf.regenerate(
    trace,
    selection as unknown as string[]
  ) as unknown as [JsTrace, number];

  const accept =
    logW === Infinity ? true : Math.log(Math.random()) < logW;

  return [accept ? proposal : trace, accept];
}

export interface Parameters {
  mu1: number;  mu2: number;
  sigma1: number; sigma2: number;
  p: number;
  proposalStdDev: number;
  numSamples: number;
  burnIn: number;
  delay: number;           // ms between animation frames
  seed: number;            // for synthetic data + initial trace
}

export interface SimulationState {
  mu1: number; mu2: number;
  acceptance_ratio: number;
  samples: { mu1: number[]; mu2: number[] };
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram:   Array<{ x: number; frequency: number }>;
}

/* make program text + synthetic data for given params */
function createModelAndData(p: Parameters) {
  const data = generate_data(
    p.mu1, p.sigma1, p.mu2, p.sigma2,
    p.p, p.numSamples, BigInt(p.seed)
  );

  const model = `(
    (sample mu1 (normal 0.0 1.0))
    (sample mu2 (normal 0.0 1.0))

    (constrain (< mu1 mu2))

    (define p ${p.p.toFixed(6)})
    (define mix (mixture (list (normal mu1 ${p.sigma1.toFixed(6)}) (normal mu2 ${p.sigma2.toFixed(6)})) (list p (- 1.0 p))))

    (define observe-point (lambda (x) (observe (gensym) mix x)))

    (for-each observe-point data)
  )`

  return { model, data };
}

/* ─────────────────────────────────────────────────────────────── main hook */

export function useSimulator() {
  /* --------------- user-tunable params ---------------------------------- */
  const [parameters, setParameters] = useState<Parameters>({
    mu1: -2, mu2: 2,
    sigma1: 1, sigma2: 1,
    p: 0.5,
    proposalStdDev: 1,
    numSamples: 1_000,
    burnIn: 100,
    delay: 1_000,
    seed: 42,
  });

  /* --------------- wasm objects / chain state --------------------------- */
  const [algorithm,   setAlgorithm]   = useState<JsGenerativeFunction | null>(null);
  const [currentTrace, setCurrentTrace] = useState<JsTrace | null>(null);
  const traceRef = useRef<JsTrace | null>(null);    // always points at latest trace

  /* derived state for charts */
  const [simState, setSimState] = useState<SimulationState | null>(null);

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

      const { model, data } = createModelAndData(parameters);
      const gf = new JsGenerativeFunction(
        model,
        { mu1: parameters.proposalStdDev, mu2: parameters.proposalStdDev },
        BigInt(parameters.seed)
      );

      /* finite-score first trace */
      const args = new Float64Array(data);
      let trace  = gf.simulate(args);
      for (let i = 0; !Number.isFinite(trace.score()) && i < 100; i++) {
        trace = gf.simulate(args);
      }

      setAlgorithm(gf);
      setCurrentTrace(trace);
      traceRef.current = trace;
      updateState(trace, [], 0);
    })().catch(console.error);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* --------------- helpers ---------------------------------------------- */
  const selection = useMemo(() => ["mu1", "mu2"] as const, []);

  const updateState = useCallback(
    (
      trace: JsTrace,
      steps: Array<{ mu1: number; mu2: number; accepted: boolean }>,
      accepted: number,
    ) => {
      const burn = Math.min(parameters.burnIn, steps.length);
      const post = steps.slice(burn);

      setSimState({
        mu1: trace.get_choice("mu1"),
        mu2: trace.get_choice("mu2"),
        acceptance_ratio: steps.length === 0 ? 0 : accepted / steps.length,
        steps,
        samples: {
          mu1: post.map(s => s.mu1),
          mu2: post.map(s => s.mu2),
        },
        distribution: [],
        histogram: [],
      });
    },
    [parameters.burnIn]
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
      const [nextTrace, accepted] = metropolisHastings(
        algorithm,
        traceRef.current,
        selection,
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
      const cap = parameters.numSamples + parameters.burnIn + 10;
      if (stepsRef.current.length > cap) stepsRef.current.shift();

      updateState(nextTrace, stepsRef.current, acceptedCountRef.current);

      if (currentStepRef.current >= parameters.numSamples + parameters.burnIn) {
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
  [
    algorithm,
    selection,
    parameters.numSamples,
    parameters.burnIn,
    pause,
    updateState,
  ]);

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

    const { model, data } = createModelAndData(parameters);
    const gf = new JsGenerativeFunction(
      model,
      { mu1: parameters.proposalStdDev, mu2: parameters.proposalStdDev },
      BigInt(parameters.seed)
    );

    const args = new Float64Array(data);
    let trace  = gf.simulate(args);
    for (let i = 0; !Number.isFinite(trace.score()) && i < 100; i++) {
      trace = gf.simulate(args);
    }

    setAlgorithm(gf);
    setCurrentTrace(trace);
    traceRef.current = trace;
    updateState(trace, [], 0);
  }, [pause, parameters, updateState]);

  /* ---------- updateParameters (depends on pause + runSimulation) ------ */
  const updateParameters = useCallback((delta: Partial<Parameters>) => {
    setParameters(prev => {
      const next = { ...prev, ...delta };

      // Validate all required parameters are valid numbers
      const requiredParams = ['mu1', 'mu2', 'sigma1', 'sigma2', 'p', 'proposalStdDev', 'numSamples', 'burnIn', 'delay', 'seed'];
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
        "proposalStdDev" in delta;

      const wasRunning = isRunningRef.current;

      if (structural) pause();

      if (structural) {
        const { model, data } = createModelAndData(next);
        const gf = new JsGenerativeFunction(
          model,
          { mu1: next.proposalStdDev, mu2: next.proposalStdDev },
          BigInt(next.seed)
        );

        const args = new Float64Array(data);
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
        updateState(trace, [], 0);

        if (wasRunning) {
          delayRef.current        = next.delay;
          lastStepTimeRef.current = performance.now();
          setIsRunning(true);
          animationId.current = requestAnimationFrame(runSimulation);
        }
      } else {
        /* non-structural change (just delay, numSamples, etc.) */
        delayRef.current = next.delay;
      }
      return next;
    });
  }, [pause, runSimulation, updateState]);

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
