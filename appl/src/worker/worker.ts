//import wasmUrl from '@/pkg/wasm_bg.wasm?url';
import {
  generate_data,
  JsGenerativeFunction,
  JsTrace,
  metropolis_hastings,
} from '../pkg/wasm';

import { Parameters } from '@/types/simulation';

/* ------------------------------------------------------------------ */
let algorithm: JsGenerativeFunction | null = null;
let trace: JsTrace | null = null;
let datasetCache: { data: Float64Array; labels: Uint8Array } | null = null;

/* -------------------- streaming control ------------------------------ */
let running = false;
let batchSize = 200; // number of proposals per batch
let delay = 0; // milliseconds between UI updates

/* total-iteration accounting */
let targetIterations = Infinity;
let iterationsDone = 0;

// Timer id for setTimeout-based pacing
let timerId: ReturnType<typeof setTimeout> | null = null;

// Compute an adaptive batch size so that slower UI rates receive smaller
// chunks, yielding smoother animation.  We aim for roughly 2 500 proposals
// per second (50 proposals every 20 ms).  Clamp to [1, 200].
function batchSizeForDelay(d: number) {
  const baseBatch = 50;
  const baseInterval = 20;
  const bs = Math.round((baseBatch * baseInterval) / Math.max(d, 20));
  return Math.min(Math.max(bs, 1), 200);
}

// Helper to stop any scheduled work
function stopTimer() {
  if (timerId !== null) {
    clearTimeout(timerId);
    timerId = null;
  }
}

function ensureInitialized(params: Parameters) {
  if (!algorithm) initialize(params);
}

function pushBatch() {
  if (!running || !algorithm || !trace) return;

  const steps: Array<{ mu1: number; mu2: number; accepted: boolean }> = [];
  let numAccepted = 0;

  const iterationsLeft = targetIterations - iterationsDone;
  const count = Math.min(batchSize, iterationsLeft);

  for (let i = 0; i < count; i++) {
    const [nextTrace, accepted] = metropolis_hastings(algorithm!, trace!, [
      'mu1',
      'mu2',
    ]);
    trace = nextTrace;
    steps.push({
      mu1: nextTrace.get_choice('mu1'),
      mu2: nextTrace.get_choice('mu2'),
      accepted,
    });
    if (accepted) numAccepted++;
    iterationsDone++;
  }

  // Emit just this batch – the UI will append.
  (self as DedicatedWorkerGlobalScope).postMessage({
    type: 'batch',
    steps,
    numAccepted,
  });

  if (iterationsDone >= targetIterations) {
    running = false;
    (self as DedicatedWorkerGlobalScope).postMessage({ type: 'done' });
    stopTimer();
    return;
  }

  timerId = setTimeout(pushBatch, delay);
}

function buildModel(p: Parameters) {
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

function initialize(params: Parameters) {
  // Generate dataset & cache
  const generated = generate_data(
    params.mu1,
    params.sigma1,
    params.mu2,
    params.sigma2,
    params.p,
    params.sampleSize,
    BigInt(params.seed)
  );

  datasetCache = {
    data: generated.data,
    labels: new Uint8Array(generated.labels),
  };

  const model = buildModel(params);

  algorithm = new JsGenerativeFunction(
    model,
    { mu1: params.proposalStdDev1, mu2: params.proposalStdDev2 },
    BigInt(params.seed)
  );

  let data = new Float64Array(datasetCache.data);

  let tr = algorithm.simulate(data);
  let attempts = 0;
  while (!Number.isFinite(tr.score()) && attempts < 1000) {
    tr = algorithm.simulate(data);
    attempts++;
  }

  if (!Number.isFinite(tr.score())) {
    throw new Error(
      'Unable to initialise Markov chain: failed to obtain finite posterior score after ' +
        attempts +
        ' attempts. Check whether the chosen parameters (e.g. very tight σ or extreme μ) are compatible with the prior.'
    );
  }

  trace = tr;

  return {
    dataset: {
      data: Array.from(datasetCache.data),
      labels: Array.from(datasetCache.labels),
    },
    mu1: trace.get_choice('mu1'),
    mu2: trace.get_choice('mu2'),
  };
}

function step() {
  if (!algorithm || !trace) throw new Error('Simulation not initialised');
  const [nextTrace, accepted] = metropolis_hastings(algorithm, trace, [
    'mu1',
    'mu2',
  ]);
  trace = nextTrace;
  return {
    mu1: nextTrace.get_choice('mu1'),
    mu2: nextTrace.get_choice('mu2'),
    accepted,
  };
}

self.onmessage = (event) => {
  const { op, args, requestId } = event.data;
  try {
    let result;
    switch (op) {
      case 'generate_data': {
        result = generate_data(
          args.mu1,
          args.sigma1,
          args.mu2,
          args.sigma2,
          args.p,
          args.n,
          BigInt(args.seed)
        );
        result = {
          data: Array.from(result.data),
          labels: Array.from(result.labels),
        };
        break;
      }
      case 'initialize': {
        result = initialize(args as Parameters);
        break;
      }
      case 'step': {
        result = step();
        break;
      }
      case 'start': {
        ensureInitialized(args.params as Parameters);

        // Update the global pacing parameters so that the very first
        // batch as well as subsequent ones respect the requested delay.
        delay = Math.max(args.params.delay ?? 0, 20);
        batchSize = batchSizeForDelay(delay);

        if (running) {
          stopTimer();
          timerId = setTimeout(pushBatch, delay);
        }

        running = true;
        stopTimer();

        targetIterations =
          args.targetIterations ??
          (args.params.burnIn || 0) + (args.params.numSteps || 0);
        iterationsDone = 0;

        // kick off first batch immediately
        pushBatch();

        result = { started: true };
        break;
      }
      case 'pause': {
        running = false;
        result = { paused: true };
        break;
      }
      case 'set_options': {
        // Allows the UI to adjust speed / batch size mid-run without resetting
        // iteration counters or touching the algorithm/trace.
        if ('delay' in args && typeof args.delay === 'number') {
          delay = args.delay;
          batchSize = batchSizeForDelay(delay);
          if (running) {
            stopTimer();
            timerId = setTimeout(pushBatch, delay);
          }
        }
        result = { updated: true };
        break;
      }
      default:
        throw new Error(`Unknown op: ${op}`);
    }
    (self as DedicatedWorkerGlobalScope).postMessage({
      ok: true,
      result,
      requestId,
    });
  } catch (err: any) {
    (self as DedicatedWorkerGlobalScope).postMessage({
      ok: false,
      error:
        err && typeof err.message === 'string'
          ? err.message
          : typeof err === 'string'
            ? err
            : JSON.stringify(err),
      requestId,
    });
  }
};

export {};
