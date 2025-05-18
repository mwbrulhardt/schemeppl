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

import { useCallback, useEffect, useRef, useState } from 'react';
import { useWasmWorker } from '@/hooks/useWorker';
import { Parameters } from '@/types/simulation';

type BatchMsg = {
  type: 'batch';
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
};
type DoneMsg = { type: 'done' };

export interface SimulationState {
  mu1: number;
  mu2: number;
  acceptance_ratio: number;
  samples: { mu1: number[]; mu2: number[] };
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
  data: number[];
  labels: number[];
}

/* ------------------------------------------------------------- the hook */
export function useSimulator() {
  /* -------------------- params (react-controlled) --------------------- */
  const [parameters, setParameters] = useState<Parameters>({
    mu1: -2,
    mu2: 2,
    sigma1: 1,
    sigma2: 1,
    p: 0.5,
    proposalStdDev1: 0.1,
    proposalStdDev2: 0.1,
    numSteps: 1_000,
    burnIn: 100,
    delay: 100,
    seed: 42,
    sampleSize: 200,
  });

  /* ------------------------ worker plumbing --------------------------- */
  const { request, on } = useWasmWorker();

  /* --------------------- internal mutable refs ------------------------ */
  const datasetRef = useRef<{ data: Float64Array; labels: Uint8Array } | null>(
    null
  );
  const stepsRef = useRef<
    Array<{ mu1: number; mu2: number; accepted: boolean }>
  >([]);
  const acceptedCountRef = useRef(0);
  const currentStepRef = useRef(0);
  const isRunningRef = useRef(false);

  /* ----------------------- react state ------------------------------- */
  const [state, setState] = useState<SimulationState | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  /* -------------- helpers ------------------------------------------- */
  const lastUiUpdateRef = useRef(0);

  const updateState = useCallback(
    (
      mu1: number,
      mu2: number,
      steps: Array<{ mu1: number; mu2: number; accepted: boolean }>,
      accepted: number,
      force = false
    ) => {
      // Throttle to ~15 fps
      const now = Date.now();
      if (!force && now - lastUiUpdateRef.current < 66) return;
      lastUiUpdateRef.current = now;

      const burn = Math.min(parameters.burnIn, steps.length);
      const post = steps.slice(burn);

      const dataset = datasetRef.current;
      if (!dataset) return;

      setState({
        mu1,
        mu2,
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
    [parameters.burnIn]
  );

  /* -------------------- initialise ------------------------------------ */
  const initialize = useCallback(
    async (p: Parameters) => {
      interface InitResult {
        dataset: { data: number[]; labels: number[] };
        mu1: number;
        mu2: number;
      }
      const res = await request<InitResult>(
        'initialize',
        p as unknown as Record<string, unknown>
      );
      const { dataset, mu1, mu2 } = res;

      datasetRef.current = {
        data: new Float64Array(dataset.data),
        labels: new Uint8Array(dataset.labels),
      };

      stepsRef.current = [];
      acceptedCountRef.current = 0;
      currentStepRef.current = 0;

      updateState(mu1, mu2, [], 0, true);
    },
    [request, updateState]
  );

  /* kick-start on mount */
  useEffect(() => {
    initialize(parameters).catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* -------------------- streaming subscription ----------------------- */
  const latestSteps = useRef<
    Array<{ mu1: number; mu2: number; accepted: boolean }>
  >([]);

  // Pause helper is needed by multiple callbacks – declare early
  const pause = useCallback(() => {
    if (!isRunningRef.current) return;
    isRunningRef.current = false;
    setIsRunning(false);
    request('pause', {});
  }, [request]);

  // Listen to batch messages from the worker
  useEffect(() => {
    const off = on<BatchMsg | DoneMsg>((msg) => {
      const m = msg as any;
      if (m && m.type === 'batch' && Array.isArray(m.steps)) {
        latestSteps.current.push(...m.steps);
      } else if (m && m.type === 'done') {
        // Worker completed requested iterations on its own.
        pause();
      }
    });
    return off;
  }, [on, pause]);

  // Flush latestSteps into React state at most once per animation frame
  useEffect(() => {
    let id: number;
    const pump = () => {
      if (latestSteps.current.length) {
        const newSteps = latestSteps.current;
        latestSteps.current = [];

        for (const s of newSteps) {
          stepsRef.current.push(s);
          if (s.accepted) acceptedCountRef.current++;
          currentStepRef.current++;
        }

        // cap memory
        const cap = parameters.numSteps + parameters.burnIn + 10;
        if (stepsRef.current.length > cap) {
          stepsRef.current.splice(0, stepsRef.current.length - cap);
        }

        const last = stepsRef.current[stepsRef.current.length - 1];
        if (last) {
          updateState(
            last.mu1,
            last.mu2,
            stepsRef.current,
            acceptedCountRef.current
          );
        }

        // Stop automatically if we've reached target iterations
        if (currentStepRef.current >= parameters.numSteps + parameters.burnIn) {
          // Ensure the UI reflects the very last batch even if we are inside
          // the 66 ms throttling window – pass `force = true` to updateState.
          const final = stepsRef.current[stepsRef.current.length - 1];
          if (final) {
            updateState(
              final.mu1,
              final.mu2,
              stepsRef.current,
              acceptedCountRef.current,
              /* force */ true
            );
          }
          pause();
        }
      }
      id = requestAnimationFrame(pump);
    };
    id = requestAnimationFrame(pump);
    return () => cancelAnimationFrame(id);
  }, [parameters.burnIn, parameters.numSteps, pause, updateState]);

  /* ------------------- controls -------------------------------------- */
  const start = useCallback(() => {
    if (isRunningRef.current) return;
    isRunningRef.current = true;
    setIsRunning(true);
    request('start', {
      params: parameters,
      delay: parameters.delay,
      targetIterations: parameters.numSteps + parameters.burnIn,
    } as unknown as Record<string, unknown>).catch(console.error);
  }, [parameters, request]);

  const reset = useCallback(() => {
    pause();
    initialize(parameters).catch(console.error);
  }, [initialize, parameters, pause]);

  const updateParameters = useCallback(
    (delta: Partial<Parameters>) => {
      setParameters((prev) => {
        const next = { ...prev, ...delta };
        // If the update touches mu1 or mu2, check that the ordering constraint
        // is satisfied. If not, just update the local state without touching the worker.
        if (('mu1' in delta || 'mu2' in delta) && next.mu1 >= next.mu2) {
          return next;
        }

        const structural =
          'mu1' in delta ||
          'mu2' in delta ||
          'sigma1' in delta ||
          'sigma2' in delta ||
          'p' in delta ||
          'seed' in delta ||
          'proposalStdDev1' in delta ||
          'proposalStdDev2' in delta ||
          'sampleSize' in delta;

        // Changing just the delay does not require a full re-initialisation,
        // but the worker still needs to pick up the new throttling interval.
        const delayChanged = 'delay' in delta;

        if (structural) {
          const wasRunning = isRunningRef.current;
          pause();
          initialize(next)
            .catch(console.error)
            .then(() => {
              if (wasRunning) {
                request('start', {
                  params: next,
                  delay: next.delay,
                  targetIterations: next.numSteps + next.burnIn,
                } as unknown as Record<string, unknown>).catch(console.error);
                isRunningRef.current = true;
                setIsRunning(true);
              }
            });
        } else if (delayChanged && isRunningRef.current) {
          // The simulation is running and only the speed has changed → inform
          // the worker without restarting the chain (this would reset the
          // internal iteration counter and cause over-sampling).  Instead send
          // a lightweight 'set_options' command.
          request('set_options', {
            delay: next.delay,
          } as unknown as Record<string, unknown>).catch(console.error);
        }
        return next;
      });
    },
    [initialize, pause, request]
  );

  /* ------------------------- api -------------------------------------- */
  return {
    state,
    parameters,
    start,
    pause,
    reset,
    updateParameters,
    isRunning,
    // algorithm is now handled inside the worker; expose boolean ready flag instead
    algorithm: state ? {} : null,
  } as const;
}
