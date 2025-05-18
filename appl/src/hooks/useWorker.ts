import { useCallback, useEffect, useRef } from 'react';

// --------------------------------------------------------------------
// Ensure that the WebAssembly worker is a **single instance** shared
// across the whole React tree.  Creating multiple workers causes the
// Wasm module to be fetched and initialised several times, which can
// leave parts of the UI waiting indefinitely for a reply from the wrong
// worker.  By hoisting the Worker instance to module scope we guarantee
// that every component using `useWasmWorker()` talks to the *same*
// worker.
// --------------------------------------------------------------------

let sharedWorker: Worker | null = null;
let workerReady = false;

// Queue of pending outbound messages posted before the worker signals it is
// ready.  Each entry stores the actual WorkerMessage as well as the
// accompanying resolve/reject callbacks so that we can set up an individual
// response listener *after* the message is finally dispatched.
const pendingQueue: Array<{
  message: WorkerMessage;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
}> = [];

interface WorkerMessage {
  op: string;
  args: Record<string, unknown>;
  requestId: string;
}

interface WorkerResponse<T = unknown> {
  ok: boolean;
  result?: T;
  error?: string;
  requestId: string;
}

/**
 * React hook that instantiates the Wasm web-worker once and provides a
 * promise-based request() helper to post messages and await their reply.
 *
 * The worker is **created exactly once** (on mount) and terminated on
 * unmount.  Each outbound request is tagged with a UUID so that concurrent
 * requests can be resolved independently.
 */
export function useWasmWorker() {
  const workerRef = useRef<Worker | null>(null);

  /* Lazily create the worker exactly once (singleton) */
  useEffect(() => {
    if (!sharedWorker) {
      sharedWorker = new Worker(
        // The "@"/worker/worker.ts" file is bundled by Next.js / Vite using the
        // `new URL` pattern.  The `type: "module"` option ensures ES modules
        // work inside the worker.
        new URL('../worker/worker.ts', import.meta.url),
        { type: 'module' }
      );

      // Listen for the one-off "ready" message from the worker.  Upon
      // receiving it flush any queued requests that were attempted while the
      // Wasm module was still loading.
      const handleReady = (ev: MessageEvent) => {
        if (
          ev.data &&
          typeof ev.data === 'object' &&
          ev.data.type === 'ready'
        ) {
          workerReady = true;
          // Flush queued messages
          for (const { message, resolve, reject } of pendingQueue) {
            // For each queued message we need to recreate the usual
            // request-response wiring.
            function makeHandler(incoming: MessageEvent<WorkerResponse>) {
              const { ok, result, error, requestId } = incoming.data;
              if (requestId !== message.requestId) return;
              sharedWorker?.removeEventListener('message', makeHandler as any);
              if (ok) resolve(result);
              else reject(new Error(error ?? 'Unknown worker error'));
            }
            sharedWorker?.addEventListener('message', makeHandler as any);
            sharedWorker?.postMessage(message);
          }
          pendingQueue.length = 0;
          sharedWorker?.removeEventListener('message', handleReady);
        }
      };
      sharedWorker.addEventListener('message', handleReady);
    }
    workerRef.current = sharedWorker;

    // DO NOT terminate the shared worker on unmount; other hooks may still
    // be using it.  If you need cleanup, consider reference counting.
    return () => {};
  }, []);

  /* ------------------------------------------------------------------ */
  const request = useCallback(
    <T = unknown>(op: string, args: Record<string, unknown>): Promise<T> => {
      return new Promise<T>((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error('Worker not initialised'));
          return;
        }

        const requestId =
          globalThis.crypto && 'randomUUID' in globalThis.crypto
            ? (globalThis.crypto as Crypto).randomUUID()
            : `${Date.now()}-${Math.random()}`;

        function handleMessage(event: MessageEvent<WorkerResponse<T>>) {
          const { ok, result, error, requestId: incomingId } = event.data;
          if (incomingId !== requestId) return; // Not ours → ignore
          workerRef.current?.removeEventListener('message', handleMessage);
          if (ok) resolve(result as T);
          else reject(new Error(error ?? 'Unknown worker error'));
        }

        const message: WorkerMessage = { op, args, requestId };

        const send = () => {
          // Attach a one-off listener for this request's response *before*
          // posting the message, to avoid missing a fast reply.
          workerRef.current?.addEventListener('message', handleMessage);
          workerRef.current?.postMessage(message);
        };

        // If the worker has not yet signalled readiness, queue the send
        // operation – it will be executed as soon as we receive the 'ready'
        // marker.
        if (!workerReady) {
          pendingQueue.push({ message, resolve, reject });
        } else {
          send();
        }
      });
    },
    []
  );

  return {
    /** Post a message to the worker and receive a promise with the result. */
    request,
    /** Subscribe to raw messages coming from the worker (those that are not
     * part of the request/response promise protocol).  Returns an unsubscribe
     * function. */
    on: <T = unknown>(listener: (data: T) => void): (() => void) => {
      if (!workerRef.current) return () => {};
      const handler = (ev: MessageEvent) => {
        // Ignore messages that belong to the request-reply protocol (they have
        // an 'ok' field).  Pass everything else to the listener.
        if (typeof ev.data === 'object' && ev.data && 'ok' in ev.data) return;
        listener(ev.data as T);
      };
      workerRef.current.addEventListener('message', handler);
      return () => workerRef.current?.removeEventListener('message', handler);
    },
  };
}
