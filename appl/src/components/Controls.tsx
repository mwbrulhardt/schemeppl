import { useCallback } from 'react';

interface Parameters {
  mean1: number;
  mean2: number;
  variance1: number;
  variance2: number;
  mixtureWeight: number;
  proposalStdDev: number;
  numSamples: number;
  burnIn: number;
  delay: number;
}

interface ControlsProps {
  parameters: Parameters;
  onUpdateParameters: (params: Partial<Parameters>) => void;
  isRunning: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
}

export default function Controls({
  parameters,
  onUpdateParameters,
  isRunning,
  onStart,
  onPause,
  onReset
}: ControlsProps) {
  const handleParameterChange = useCallback((key: keyof Parameters, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      onUpdateParameters({ [key]: numValue });
    }
  }, [onUpdateParameters]);

  return (
    <div className="bg-gray-50 p-4 rounded-lg mb-8">
      <h2 className="text-xl font-semibold mb-4">Parameters</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Component 1 Mean</label>
          <input
            type="number"
            value={parameters.mean1}
            onChange={(e) => handleParameterChange('mean1', e.target.value)}
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 2 Mean</label>
          <input
            type="number"
            value={parameters.mean2}
            onChange={(e) => handleParameterChange('mean2', e.target.value)}
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 1 Variance</label>
          <input
            type="number"
            value={parameters.variance1}
            onChange={(e) => handleParameterChange('variance1', e.target.value)}
            min="0.1"
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 2 Variance</label>
          <input
            type="number"
            value={parameters.variance2}
            onChange={(e) => handleParameterChange('variance2', e.target.value)}
            min="0.1"
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Mixture Weight (π)</label>
          <input
            type="number"
            value={parameters.mixtureWeight}
            onChange={(e) => handleParameterChange('mixtureWeight', e.target.value)}
            min="0.1"
            max="0.9"
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Proposal StdDev</label>
          <input
            type="number"
            value={parameters.proposalStdDev}
            onChange={(e) => handleParameterChange('proposalStdDev', e.target.value)}
            min="0.1"
            step="0.1"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Number of Samples</label>
          <input
            type="number"
            value={parameters.numSamples}
            onChange={(e) => handleParameterChange('numSamples', e.target.value)}
            min="100"
            step="100"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Burn-in Period</label>
          <input
            type="number"
            value={parameters.burnIn}
            onChange={(e) => handleParameterChange('burnIn', e.target.value)}
            min="0"
            step="10"
            className="w-full p-2 border rounded"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">
            Simulation Speed (ms): {parameters.delay}
          </label>
          <input
            type="range"
            value={parameters.delay}
            onChange={(e) => handleParameterChange('delay', e.target.value)}
            min="1"
            max="100"
            className="w-full"
          />
        </div>
      </div>
      <div className="mt-4 flex gap-4">
        <button
          onClick={isRunning ? onPause : onStart}
          className={`px-4 py-2 rounded text-white ${
            isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
          }`}
        >
          {isRunning ? 'Pause' : 'Start'}
        </button>
        <button
          onClick={onReset}
          className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300"
        >
          Reset
        </button>
      </div>
    </div>
  );
} 