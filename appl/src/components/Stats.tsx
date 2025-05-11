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

interface SimulationState {
  current_x: number;
  proposed_x: number;
  acceptance_ratio: number;
  samples: number[];
  steps: Array<{ x: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
}

interface StatsProps {
  state: SimulationState | null;
  parameters: Parameters;
}

export default function Stats({ state, parameters }: StatsProps) {
  if (!state) return null;

  const midpoint = (parameters.mean1 + parameters.mean2) / 2;
  const comp1Samples = state.samples.filter(x => x < midpoint);
  const comp2Samples = state.samples.filter(x => x >= midpoint);

  const comp1Mean = comp1Samples.length > 0
    ? comp1Samples.reduce((a, b) => a + b, 0) / comp1Samples.length
    : NaN;

  const comp2Mean = comp2Samples.length > 0
    ? comp2Samples.reduce((a, b) => a + b, 0) / comp2Samples.length
    : NaN;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Current Step</div>
        <div className="text-lg font-semibold">{state.steps.length}</div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Samples Collected</div>
        <div className="text-lg font-semibold">{state.samples.length}</div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Acceptance Ratio</div>
        <div className="text-lg font-semibold">
          {(state.acceptance_ratio * 100).toFixed(2)}%
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Component 1 Mean</div>
        <div className="text-lg font-semibold">
          {isNaN(comp1Mean) ? 'N/A' : comp1Mean.toFixed(4)}
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Component 2 Mean</div>
        <div className="text-lg font-semibold">
          {isNaN(comp2Mean) ? 'N/A' : comp2Mean.toFixed(4)}
        </div>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="text-sm font-medium text-gray-500">Target Parameters</div>
        <div className="text-sm">
          μ₁={parameters.mean1.toFixed(2)}, μ₂={parameters.mean2.toFixed(2)}
          <br />
          σ₁²={parameters.variance1.toFixed(2)}, σ₂²={parameters.variance2.toFixed(2)}
          <br />
          π={parameters.mixtureWeight.toFixed(2)}
        </div>
      </div>
    </div>
  );
} 