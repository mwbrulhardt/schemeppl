'use client';

// (no state hooks needed at this level)
import AlgorithmDescription from '@/components/AlgorithmDescription';
import Charts from '@/components/Charts';
import Controls from '@/components/Controls';
import WalkVisualization from '@/components/WalkVisualization';
import { Chart, registerables } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { useSimulator } from '../hooks/useSimulator';

// Register Chart.js plugins
Chart.register(...registerables, annotationPlugin);

export default function Home() {
  // No explicit error handling yet – hook prints to console for now

  const {
    isRunning,
    start,
    pause,
    reset,
    updateParameters,
    state,
    parameters,
  } = useSimulator();

  const isLoading = state === null;

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
