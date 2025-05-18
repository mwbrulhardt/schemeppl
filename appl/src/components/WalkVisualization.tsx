'use client';

import {
  Chart,
  ScatterController,
  BubbleController,
  PointElement,
  LineElement,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { useMemo } from 'react';
import { SimulationState } from '@/hooks/useSimulator';
import type { Parameters } from '@/types/simulation';
import { zoomAndPanPlugin } from '@/lib/chart/zoomAndPanPlugin';

// Register core elements + our custom plugin **once**
Chart.register(
  ScatterController,
  BubbleController,
  PointElement,
  LineElement,
  LinearScale,
  Tooltip,
  Legend,
  Filler
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
  /** Number of rejected proposals before moving to the next accepted location */
  rejects?: number;
}

export default function WalkVisualization({
  state,
  parameters,
  isRunning,
}: Props) {
  /* --------------------------------------------------------------- */
  /* 1. Build datasets                                               */
  /* --------------------------------------------------------------- */
  const { datasets, minX, maxX, minY, maxY } = useMemo(() => {
    const steps = state?.steps ?? [];

    // Pre-compute indices of accepted points to measure the number of
    // rejected proposals between them.  For each accepted point we attach a
    // `rejects` property indicating how many proposals were rejected before
    // the chain moved to the *next* accepted location.
    const acceptedIndices: number[] = [];
    steps.forEach((s, idx) => {
      if (s.accepted) acceptedIndices.push(idx);
    });

    const acceptedPts: DataPoint[] = [];
    const rejectedPts: DataPoint[] = [];

    const acceptedRadii: number[] = [];
    const acceptedBgColors: string[] = [];

    acceptedIndices.forEach((accIdx, k) => {
      const nextAccIdx =
        k + 1 < acceptedIndices.length ? acceptedIndices[k + 1] : steps.length;
      const rejectsBetween = Math.max(nextAccIdx - accIdx - 1, 0);

      const p = steps[accIdx];
      acceptedPts.push({
        x: p.mu1,
        y: p.mu2,
        step: accIdx + 1,
        rejects: rejectsBetween,
      });

      // Radius: base 2px + 0.8 px per rejected proposal (clamped)
      acceptedRadii.push(Math.min(2 + rejectsBetween * 0.8, 10));

      // Opacity inversely proportional to rejected count (≥ 0.3, ≤ 1.0)
      const alpha = Math.max(0.3, 1 - rejectsBetween * 0.1);
      acceptedBgColors.push(`rgba(77, 77, 77, ${alpha})`);
    });

    // Rejected points are plotted too but we do not need extra tooltip info.
    steps.forEach((s, idx) => {
      if (!s.accepted) {
        rejectedPts.push({ x: s.mu1, y: s.mu2, step: idx + 1 });
      }
    });

    const axisPad = 2;

    // Collect all x (mu1) and y (mu2) values currently in view, including
    // the original parameter means as sensible fallbacks when there are no
    // samples yet.
    const xs = [parameters.mu1, parameters.mu2, ...steps.map((s) => s.mu1)];
    const ys = [parameters.mu1, parameters.mu2, ...steps.map((s) => s.mu2)];

    const minX = Math.min(...xs) - axisPad;
    const maxX = Math.max(...xs) + axisPad;
    const minY = Math.min(...ys) - axisPad;
    const maxY = Math.max(...ys) + axisPad;

    return {
      datasets: [
        {
          label: 'Accepted',
          data: acceptedPts,
          backgroundColor: acceptedBgColors,
          borderColor: '#4d4d4d',
          borderWidth: 1,
          pointRadius: acceptedRadii,
          pointHoverRadius: acceptedRadii,
        },
        {
          label: 'Rejected',
          data: rejectedPts,
          backgroundColor: 'rgba(215,48,39,0.7)',
          pointRadius: 2,
          pointHoverRadius: 4,
        },
      ],
      minX,
      maxX,
      minY,
      maxY,
    };
  }, [state, parameters]);

  /* --------------------------------------------------------------- */
  /* 2. Chart options                                                */
  /* --------------------------------------------------------------- */
  const options = useMemo(
    () =>
      ({
        responsive: true,
        maintainAspectRatio: false,
        animation: isRunning ? false : { duration: 300 },
        interaction: {
          mode: 'point' as const,
          intersect: true,
        },
        scales: {
          x: {
            type: 'linear' as const,
            minX,
            maxX,
            title: { display: true, text: 'μ₁' },
            grid: { color: '#e0e0e0' },
          },
          y: {
            type: 'linear' as const,
            minY,
            maxY,
            title: { display: true, text: 'μ₂' },
            grid: { color: '#e0e0e0' },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      }) as ChartOptions<'scatter'>,
    [isRunning, minX, maxX, minY, maxY]
  );

  /* --------------------------------------------------------------- */
  /* 3. Render                                                       */
  /* --------------------------------------------------------------- */
  return (
    <div className="relative h-[600px] w-full max-w-4xl mx-auto bg-white rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4">
        Metropolis-Hastings Random Walk (including burn-in)
      </h2>
      <Scatter
        data={{ datasets }}
        options={options}
        plugins={[zoomAndPanPlugin]}
      />
      <div className="absolute bottom-2 right-2 text-xs text-gray-500 bg-white/80 px-2 py-1 rounded">
        <span className="mr-2">🖱️ Scroll to zoom</span>
        <span>🖐️ Drag to pan</span>
      </div>
    </div>
  );
}
