import { SimulationState } from '@/hooks/useSimulator';
import React from 'react';
import { calculateMean, calculateStdDev } from '@/utils/stats';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';

interface StatisticsProps {
  state: SimulationState | null;
}

const Statistics: React.FC<StatisticsProps> = ({ state }) => {
  // Determine whether the simulation has produced any samples yet.
  // We treat the simulation as "not started" when we either don't have a
  // state object yet or the `steps` array is still empty. In that case we
  // fall back to displaying zeros for all statistics.

  const hasStarted = Boolean(state && state.steps.length > 0);

  const mu1Value = hasStarted ? calculateMean(state!.samples.mu1) : 0;
  const mu2Value = hasStarted ? calculateMean(state!.samples.mu2) : 0;

  const mu1StdDev = hasStarted ? calculateStdDev(state!.samples.mu1) : 0;
  const mu2StdDev = hasStarted ? calculateStdDev(state!.samples.mu2) : 0;

  const acceptancePercentage = ((state?.acceptance_ratio ?? 0) * 100).toFixed(
    1
  );

  return (
    <div className="bg-white rounded-lg shadow p-5 h-full flex flex-col space-y-8">
      <h2 className="text-xl font-semibold">Statistics</h2>

      {/* Parameters Group */}
      <section className="flex flex-col space-y-4">
        <h3 className="text-sm font-medium text-gray-500">Parameters</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 px-4 py-3 rounded text-center">
            <p className="text-xs tracking-wide text-gray-600 mb-1">
              <InlineMath math={'\\hat{\\mu}_1'} />
            </p>
            <p className="text-lg font-semibold">{mu1Value.toFixed(3)}</p>
            <p className="text-[10px] text-gray-500">
              σ = {mu1StdDev.toFixed(3)}
            </p>
          </div>
          <div className="bg-gray-50 px-4 py-3 rounded text-center">
            <p className="text-xs tracking-wide text-gray-600 mb-1">
              <InlineMath math={'\\hat{\\mu}_2'} />
            </p>
            <p className="text-lg font-semibold">{mu2Value.toFixed(3)}</p>
            <p className="text-[10px] text-gray-500">
              σ = {mu2StdDev.toFixed(3)}
            </p>
          </div>
        </div>
      </section>

      {/* Diagnostics Group */}
      <section className="flex flex-col space-y-4">
        <h3 className="text-sm font-medium text-gray-500">Diagnostics</h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 px-4 py-3 rounded text-center">
            <p className="text-xs uppercase tracking-wide text-gray-600 mb-1">
              Acceptance&nbsp;Rate
            </p>
            <p className="text-lg font-semibold">{acceptancePercentage}%</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Statistics;
