import { SimulationState } from '@/hooks/useSimulator';
import React from 'react';

interface StatisticsProps {
  state: SimulationState | null;
}

const Statistics: React.FC<StatisticsProps> = ({ state }) => {
  if (!state) return null;

  // Calculate standard deviations from samples
  const calculateStdDev = (samples: number[]) => {
    if (samples.length === 0) return 0;
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / samples.length;
    return Math.sqrt(variance);
  };

  const mu1StdDev = calculateStdDev(state.samples.mu1);
  const mu2StdDev = calculateStdDev(state.samples.mu2);

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-8">
      <h2 className="text-xl font-semibold mb-4">Simulation Statistics</h2>
      
      {/* Parameters Group */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-500 mb-3">Current Parameters</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">μ₁</p>
            <p className="text-lg font-medium">{state.mu1.toFixed(3)}</p>
            <p className="text-xs text-gray-500">σ: {mu1StdDev.toFixed(3)}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">μ₂</p>
            <p className="text-lg font-medium">{state.mu2.toFixed(3)}</p>
            <p className="text-xs text-gray-500">σ: {mu2StdDev.toFixed(3)}</p>
          </div>
        </div>
      </div>

      {/* Sampling Progress Group */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-500 mb-3">Sampling Progress</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Total Steps</p>
            <p className="text-lg font-medium">{state.steps.length}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Samples Collected</p>
            <p className="text-lg font-medium">{state.samples.mu1.length}</p>
          </div>
        </div>
      </div>

      {/* Acceptance Metrics Group */}
      <div>
        <h3 className="text-sm font-medium text-gray-500 mb-3">Acceptance Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Acceptance Rate</p>
            <p className="text-lg font-medium">{(state.acceptance_ratio * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics; 