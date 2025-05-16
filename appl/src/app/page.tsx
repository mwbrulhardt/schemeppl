'use client';

import AlgorithmDescription from '@/components/AlgorithmDescription';
import Charts from '@/components/Charts';
import Controls from '@/components/Controls';
import Statistics from '@/components/Statistics';
import WalkVisualization from '@/components/WalkVisualization';
import { Chart, registerables } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { useEffect, useState } from 'react';
import { useSimulator } from '../hooks/useSimulator';

// Register Chart.js plugins
Chart.register(...registerables, annotationPlugin);

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const [error] = useState<string | null>(null);
  
  const {
    algorithm,
    isRunning,
    start,
    pause,
    reset,
    updateParameters,
    state,
    parameters
  } = useSimulator();

  useEffect(() => {
    if (algorithm) {
      setIsLoading(false);
    }
  }, [algorithm]);

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
          <code className="bg-gray-100 p-1 rounded">wasm-pack build --target web</code>
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
        onUpdateParameters={updateParameters}
        isRunning={isRunning}
        onStart={start}
        onPause={pause}
        onReset={reset}
      />

      <Statistics state={state} />

      <div className="space-y-8 w-full">

        {/* Charts in Grid Layout */}{/* Charts in Grid Layout */}
        <Charts state={state} parameters={parameters} />
        
        {/* Walk Visualization (Full Width) */}
        {state && <WalkVisualization state={state} isRunning={isRunning} parameters={parameters} />}

        
      </div>

      
    </main>
  );
}
