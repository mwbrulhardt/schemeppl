import { useCallback, useEffect, useRef } from 'react';

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

  /* Lazily create the worker exactly once */
  useEffect(() => {
    workerRef.current = new Worker(
      // The "@"/worker/worker.ts" file is bundled by Next.js / Vite using the
      // `new URL` pattern.  The `type: "module"` option ensures ES modules
      // work inside the worker.
      new URL('../worker/worker.ts', import.meta.url),
      { type: 'module' }
    );
    return () => {
      workerRef.current?.terminate();
    };
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

        workerRef.current.addEventListener('message', handleMessage);
        const message: WorkerMessage = { op, args, requestId };
        workerRef.current.postMessage(message);
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
