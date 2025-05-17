'use client';

import {
  Chart,
  ScatterController,
  PointElement,
  LineElement,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
  TooltipItem,
  ChartOptions,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { useMemo } from 'react';
import type { Parameters, SimulationState } from '@/hooks/useSimulator';
import { zoomAndPanPlugin } from '@/lib/chart/zoomAndPanPlugin';

// Register core elements + our custom plugin **once**
Chart.register(
    ScatterController,
    PointElement,
    LineElement,
    LinearScale,
    Tooltip,
    Legend,
    Filler,
);

interface Props {
  state: SimulationState | null;
  parameters: Parameters;
  isRunning: boolean;
}

// Define the data point type with step property
interface DataPoint {
  x: number;
  y: number;
  step: number;
}

export default function WalkVisualization({
  state,
  parameters,
  isRunning,
}: Props) {
  /* --------------------------------------------------------------- */
  /* 1. Build datasets                                               */
  /* --------------------------------------------------------------- */
  const { datasets, min, max } = useMemo(() => {
    const steps = state?.steps ?? [];

    const accepted = steps.filter((s) => s.accepted);
    const rejected = steps.filter((s) => !s.accepted);

    const toPts = (arr: typeof steps) =>
      arr.map((p, i) => ({ x: p.mu1, y: p.mu2, step: i + 1 }));

    const acceptedPts = toPts(accepted);
    const rejectedPts = toPts(rejected);

    const axisPad = 2;
    const { mu1, mu2 } = parameters;
    const lo = Math.min(mu1, mu2) - axisPad;
    const hi = Math.max(mu1, mu2) + axisPad;

    return {
      datasets: [
        // thin grey path through accepted points
        {
          label: 'Path',
          data: acceptedPts,
          showLine: true,
          pointRadius: 0,
          borderWidth: 1,
          borderColor: '#999',
        },
        {
          label: 'Accepted',
          data: acceptedPts,
          backgroundColor: '#4d4d4d',
          pointRadius: 3,
          pointHoverRadius: 6,
        },
        {
          label: 'Rejected',
          data: rejectedPts,
          backgroundColor: 'rgba(215,48,39,0.7)',
          pointRadius: 3,
          pointHoverRadius: 6,
        },
      ],
      min: lo,
      max: hi,
    };
  }, [state, parameters]);

  /* --------------------------------------------------------------- */
  /* 2. Chart options                                                */
  /* --------------------------------------------------------------- */
  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: isRunning ? false : { duration: 300 },
      interaction: { 
        mode: 'point' as const, 
        intersect: true 
      },
      scales: {
        x: {
          type: 'linear' as const,
          min,
          max,
          title: { display: true, text: 'μ₁' },
          grid: { color: '#e0e0e0' },
        },
        y: {
          type: 'linear' as const,
          min,
          max,
          title: { display: true, text: 'μ₂' },
          grid: { color: '#e0e0e0' },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          position: 'nearest',
          callbacks: {
            title() {
              return '';  // No title needed
            },
            label(ctx: TooltipItem<'scatter'>) {
              const { x, y } = ctx.parsed;
              const step = (ctx.raw as DataPoint)?.step ?? '?';
              return `#${step} ${ctx.dataset.label}: (${x.toFixed(2)}, ${y.toFixed(2)})`;
            },
          },
        },
      },
    } as ChartOptions<'scatter'>),
    [isRunning, min, max],
  );

  /* --------------------------------------------------------------- */
  /* 3. Render                                                       */
  /* --------------------------------------------------------------- */
  return (
    <div className="relative h-[600px] w-full max-w-4xl mx-auto bg-white rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4">Metropolis-Hastings Random Walk</h2>
      <Scatter data={{ datasets }} options={options} plugins={[zoomAndPanPlugin]}/>
      <div className="absolute bottom-2 right-2 text-xs text-gray-500 bg-white/80 px-2 py-1 rounded">
        <span className="mr-2">🖱️ Scroll to zoom</span>
        <span>🖐️ Drag to pan</span>
      </div>
    </div>
  );
}
