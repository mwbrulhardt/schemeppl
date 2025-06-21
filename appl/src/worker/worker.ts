//import wasmUrl from '@/pkg/wasm_bg.wasm?url';
import {
  generate_data,
  JsGenerativeFunction,
  JsTrace,
  JsRng,
  metropolis_hastings_with_proposal_js,
} from '../pkg/wasm';

import { Parameters } from '@/types/simulation';

/* ------------------------------------------------------------------ */
let algorithm: JsGenerativeFunction | null = null;
let proposalFunction: JsGenerativeFunction | null = null;
let trace: JsTrace | null = null;
let rng: JsRng | null = null;
let datasetCache: { data: Float64Array; labels: Uint8Array } | null = null;

// Store proposal step sizes globally
let proposalStepSize1 = 0.15;
let proposalStepSize2 = 0.15;

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
  if (!running || !algorithm || !trace || !rng || !proposalFunction) return;

  const steps: Array<{ mu1: number; mu2: number; accepted: boolean }> = [];
  let numAccepted = 0;

  const iterationsLeft = targetIterations - iterationsDone;
  const count = Math.min(batchSize, iterationsLeft);

  for (let i = 0; i < count; i++) {
    const stepResult = step();
    steps.push({
      mu1: stepResult.mu1,
      mu2: stepResult.mu2,
      accepted: stepResult.accepted,
    });
    if (stepResult.accepted) numAccepted++;
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

function buildProposal() {
  return `(
    (sample mu1 (normal current_mu1 step_size1))
    (sample mu2 (normal current_mu2 step_size2))
    #t
  )`;
}

function initialize(params: Parameters) {
  // Initialize RNG
  rng = new JsRng(BigInt(params.seed));

  // Store proposal step sizes
  proposalStepSize1 = params.proposalStdDev1;
  proposalStepSize2 = params.proposalStdDev2;

  // Generate dataset & cache
  const generated = generate_data(
    params.mu1,
    params.sigma1,
    params.mu2,
    params.sigma2,
    params.p,
    params.sampleSize,
    rng
  );

  datasetCache = {
    data: generated.data,
    labels: new Uint8Array(generated.labels),
  };

  const model = buildModel(params);
  const proposal = buildProposal();

  // Create the main generative function
  algorithm = new JsGenerativeFunction(['data'], model);

  // Create the proposal function
  proposalFunction = new JsGenerativeFunction(
    ['current_mu1', 'current_mu2', 'step_size1', 'step_size2'],
    proposal
  );

  let data = new Float64Array(datasetCache.data);

  let tr = algorithm.simulate(rng, data);
  let attempts = 0;
  while (!Number.isFinite(tr.score()) && attempts < 1000) {
    tr = algorithm.simulate(rng, data);
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
  if (!algorithm || !trace || !rng || !proposalFunction)
    throw new Error('Simulation not initialised');

  // Get current values from the trace
  const currentMu1 = trace.get_choice('mu1');
  const currentMu2 = trace.get_choice('mu2');

  // Use stored step sizes
  const stepSize1 = proposalStepSize1;
  const stepSize2 = proposalStepSize2;

  // Create proposal arguments
  const proposalArgs = [currentMu1, currentMu2, stepSize1, stepSize2];
  const proposalArgsArray = new Array(proposalArgs.length);
  for (let i = 0; i < proposalArgs.length; i++) {
    proposalArgsArray[i] = proposalArgs[i];
  }

  const result = metropolis_hastings_with_proposal_js(
    rng,
    trace,
    proposalFunction,
    proposalArgsArray,
    false, // check
    null // observations
  );

  const [nextTrace, accepted] = result;
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
        // Create a temporary RNG for this operation
        const tempRng = new JsRng(BigInt(args.seed));
        result = generate_data(
          args.mu1,
          args.sigma1,
          args.mu2,
          args.sigma2,
          args.p,
          args.n,
          tempRng
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

/* -------------------------------------------------------------- */
// Signal to the main thread that the worker (and therefore the Wasm module)
// has been fully loaded.  Messages queued by the UI will be flushed once this
// notification is received.
(self as DedicatedWorkerGlobalScope).postMessage({ type: 'ready' });

export {};
