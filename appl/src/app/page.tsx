'use client';

import { useEffect, useState, useCallback } from 'react';
import AlgorithmDescription from '@/components/AlgorithmDescription';
import Charts from '@/components/Charts';
import Controls from '@/components/Controls';
import WalkVisualization from '@/components/WalkVisualization';
import { Chart, registerables } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { useSimulator } from '../hooks/useSimulator';
import { useWasmWorker } from '@/hooks/useWorker';

// Register Chart.js plugins
Chart.register(...registerables, annotationPlugin);

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { request } = useWasmWorker();

  const runGenerateData = useCallback(() => {
    setIsLoading(true);
    setError(null);
    request('generate_data', {
      mu1: 0,
      sigma1: 1,
      mu2: 5,
      sigma2: 2,
      p: 0.5,
      n: 100,
      seed: 42,
    })
      .then(() => {
        setIsLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setIsLoading(false);
      });
  }, [request]);

  const {
    isRunning,
    start,
    pause,
    reset,
    updateParameters,
    state,
    parameters,
  } = useSimulator();

  useEffect(() => {
    runGenerateData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (state) {
      setIsLoading(false);
    }
  }, [state]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-lg">Loading WebAssembly...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-red-500">
        <h2 className="font-bold">Error loading WebAssembly</h2>
        <p>{error}</p>
        <p className="mt-2">
          Please make sure you&apos;ve built the WASM module with{' '}
          <code className="bg-gray-100 p-1 rounded">
            wasm-pack build --target web
          </code>
          and that you&apos;re running this page from a web server.
        </p>
      </div>
    );
  }

  return (
    <main className="max-w-7xl mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-8">
        Gaussian Mixture Model - Metropolis-Hastings Sampling
      </h1>
      <AlgorithmDescription />

      <Controls
        parameters={parameters}
        state={state}
        onUpdateParameters={updateParameters}
        isRunning={isRunning}
        onStart={start}
        onPause={pause}
        onReset={reset}
      />

      <div className="space-y-8 w-full">
        <Charts state={state} parameters={parameters} />
        {state && (
          <WalkVisualization
            state={state}
            isRunning={isRunning}
            parameters={parameters}
          />
        )}
      </div>
    </main>
  );
}
