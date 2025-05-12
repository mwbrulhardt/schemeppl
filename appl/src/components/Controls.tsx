import React, { useCallback, useState, useEffect } from 'react';

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
  // Local state for input values and validity
  const [inputValues, setInputValues] = useState<Record<keyof Parameters, string>>({
    mean1: parameters.mean1.toString(),
    mean2: parameters.mean2.toString(),
    variance1: parameters.variance1.toString(),
    variance2: parameters.variance2.toString(),
    mixtureWeight: parameters.mixtureWeight.toString(),
    proposalStdDev: parameters.proposalStdDev.toString(),
    numSamples: parameters.numSamples.toString(),
    burnIn: parameters.burnIn.toString(),
    delay: parameters.delay.toString()
  });
  const [inputValidity, setInputValidity] = useState<Record<keyof Parameters, boolean>>({
    mean1: true,
    mean2: true,
    variance1: true,
    variance2: true,
    mixtureWeight: true,
    proposalStdDev: true,
    numSamples: true,
    burnIn: true,
    delay: true
  });

  // Validation rules for each parameter
  const validators: Record<keyof Parameters, (value: string) => boolean> = {
    mean1: v => !isNaN(Number(v)),
    mean2: v => !isNaN(Number(v)),
    variance1: v => !isNaN(Number(v)) && Number(v) > 0,
    variance2: v => !isNaN(Number(v)) && Number(v) > 0,
    mixtureWeight: v => !isNaN(Number(v)) && Number(v) > 0 && Number(v) < 1,
    proposalStdDev: v => !isNaN(Number(v)) && Number(v) > 0,
    numSamples: v => !isNaN(Number(v)) && Number(v) >= 100,
    burnIn: v => !isNaN(Number(v)) && Number(v) >= 0,
    delay: v => !isNaN(Number(v)) && Number(v) >= 0 && Number(v) <= 1000
  };

  // Update local state when parameters change
  useEffect(() => {
    setInputValues({
      mean1: parameters.mean1.toString(),
      mean2: parameters.mean2.toString(),
      variance1: parameters.variance1.toString(),
      variance2: parameters.variance2.toString(),
      mixtureWeight: parameters.mixtureWeight.toString(),
      proposalStdDev: parameters.proposalStdDev.toString(),
      numSamples: parameters.numSamples.toString(),
      burnIn: parameters.burnIn.toString(),
      delay: parameters.delay.toString()
    });
    setInputValidity({
      mean1: true,
      mean2: true,
      variance1: true,
      variance2: true,
      mixtureWeight: true,
      proposalStdDev: true,
      numSamples: true,
      burnIn: true,
      delay: true
    });
  }, [parameters]);

  const handleParameterChange = useCallback((key: keyof Parameters, value: string) => {
    setInputValues(prev => ({ ...prev, [key]: value }));
    const isValid = validators[key](value);
    setInputValidity(prev => ({ ...prev, [key]: isValid }));
    if (isValid) {
      onUpdateParameters({ [key]: parseFloat(value) });
    }
  }, [onUpdateParameters]);

  // Helper to get input class
  const getInputClass = (key: keyof Parameters) =>
    `w-full p-2 border rounded ${inputValidity[key] ? '' : 'border-red-500 ring-2 ring-red-200'}`;

  return (
    <div className="bg-gray-50 p-4 rounded-lg mb-8">
      <h2 className="text-xl font-semibold mb-4">Parameters</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Component 1 Mean</label>
          <input
            type="number"
            value={inputValues.mean1}
            onChange={(e) => handleParameterChange('mean1', e.target.value)}
            step="0.1"
            className={getInputClass('mean1')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 2 Mean</label>
          <input
            type="number"
            value={inputValues.mean2}
            onChange={(e) => handleParameterChange('mean2', e.target.value)}
            step="0.1"
            className={getInputClass('mean2')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 1 Variance</label>
          <input
            type="number"
            value={inputValues.variance1}
            onChange={(e) => handleParameterChange('variance1', e.target.value)}
            min="0.1"
            step="0.1"
            className={getInputClass('variance1')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Component 2 Variance</label>
          <input
            type="number"
            value={inputValues.variance2}
            onChange={(e) => handleParameterChange('variance2', e.target.value)}
            min="0.1"
            step="0.1"
            className={getInputClass('variance2')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Mixture Weight (π)</label>
          <input
            type="number"
            value={inputValues.mixtureWeight}
            onChange={(e) => handleParameterChange('mixtureWeight', e.target.value)}
            min="0.01"
            max="0.99"
            step="0.01"
            className={getInputClass('mixtureWeight')}
            disabled={isRunning}
          />
          {!inputValidity.mixtureWeight && (
            <div className="text-xs text-red-600 mt-1">Enter a value between 0 and 1 (exclusive).</div>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Proposal StdDev</label>
          <input
            type="number"
            value={inputValues.proposalStdDev}
            onChange={(e) => handleParameterChange('proposalStdDev', e.target.value)}
            min="0.1"
            step="0.1"
            className={getInputClass('proposalStdDev')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Number of Samples</label>
          <input
            type="number"
            value={inputValues.numSamples}
            onChange={(e) => handleParameterChange('numSamples', e.target.value)}
            min="100"
            step="100"
            className={getInputClass('numSamples')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Burn-in Period</label>
          <input
            type="number"
            value={inputValues.burnIn}
            onChange={(e) => handleParameterChange('burnIn', e.target.value)}
            min="0"
            step="10"
            className={getInputClass('burnIn')}
            disabled={isRunning}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">
            Simulation Speed (ms): {parameters.delay}
          </label>
          <input
            type="range"
            value={inputValues.delay}
            onChange={(e) => handleParameterChange('delay', e.target.value)}
            min="0"
            max="1000"
            step="10"
            className={getInputClass('delay')}
            disabled={isRunning}
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Fast (0ms)</span>
            <span>Slow (1000ms)</span>
          </div>
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