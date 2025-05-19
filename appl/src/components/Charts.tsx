/* eslint-disable @typescript-eslint/no-explicit-any */
import { Parameters } from '@/types/simulation';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import annotationPlugin, {
  AnnotationOptions,
  AnnotationTypeRegistry,
} from 'chartjs-plugin-annotation';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { calculateMean, kde, normalPdf } from '@/utils/stats';
import Statistics from '@/components/Statistics';
import { SimulationState } from '@/hooks/useSimulator';
Chart.register(...registerables, annotationPlugin);

/*****************************
 * Types
 *****************************/
interface ChartsProps {
  state: SimulationState | null;
  parameters: Parameters;
}

type AnnotationConfig = {
  [key: string]: AnnotationOptions<keyof AnnotationTypeRegistry>;
};

/*****************************
 * Generic helpers
 *****************************/

/** Build a simple line‑style dataset */
const lineDataset = (
  label: string,
  color: string,
  dashed = false,
  yAxisID = 'y'
) => ({
  label,
  borderColor: color,
  borderWidth: 2,
  pointRadius: 0,
  fill: false,
  borderDash: dashed ? [2, 2] : undefined,
  data: [] as { x: number; y: number }[],
  yAxisID,
});

/** Shared chart options */
const baseOptions = (
  title: string,
  xLabel: string,
  yLabel: string,
  extra: Record<string, unknown> = {}
) => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: { display: true, text: title },
    tooltip: { mode: 'index' as const, intersect: false },
    annotation: { annotations: {} },
  },
  scales: {
    x: { type: 'linear' as const, title: { display: true, text: xLabel } },
    y: { title: { display: true, text: yLabel } },
    ...extra,
  },
  animation: { duration: 0 } as const,
});

/*****************************
 * React component
 *****************************/

export default function Charts({ state, parameters }: ChartsProps) {
  /***** Refs *****/
  const distributionRef = useRef<Chart | null>(null);
  const mu1TraceRef = useRef<Chart | null>(null);
  const mu2TraceRef = useRef<Chart | null>(null);

  /***** Derived helpers *****/
  const xDomain = useCallback(() => {
    const { mu1, mu2, sigma1, sigma2 } = parameters;

    // Calculate theoretical range
    const theoreticalMin = Math.min(mu1, mu2) - 2 * Math.max(sigma1, sigma2);
    const theoreticalMax = Math.max(mu1, mu2) + 2 * Math.max(sigma1, sigma2);

    // If we have data, use it to adjust the range
    if (state?.data && state.data.length > 0) {
      const dataMin = Math.min(...state.data);
      const dataMax = Math.max(...state.data);
      const dataRange = dataMax - dataMin;

      // Use the tighter of the two ranges, with a small buffer
      return {
        xMin: Math.min(theoreticalMin, dataMin - 0.1 * dataRange),
        xMax: Math.max(theoreticalMax, dataMax + 0.1 * dataRange),
      } as const;
    }

    // Fallback to theoretical range if no data
    return {
      xMin: theoreticalMin,
      xMax: theoreticalMax,
    } as const;
  }, [parameters, state?.data]);

  const numPoints = 100;
  const xVals = useMemo(() => {
    const { xMin, xMax } = xDomain();
    return Array.from(
      { length: numPoints },
      (_, i) => xMin + ((xMax - xMin) * i) / (numPoints - 1)
    );
  }, [xDomain, numPoints]);

  /***** Chart initialisation helpers *****/
  const initChart = (
    id: string,
    chartRef: React.MutableRefObject<Chart | null>,
    configFactory: () => ChartConfiguration
  ) => {
    const ctx = document.getElementById(id) as HTMLCanvasElement | null;
    if (ctx && !chartRef.current)
      chartRef.current = new Chart(ctx, configFactory());
  };

  /***** Initialise charts (once) *****/
  useEffect(() => {
    // Only initialize charts when we have state data
    if (!state) return;

    /* μ₁ trace */
    initChart('component1-trace-chart', mu1TraceRef, () => ({
      type: 'line',
      data: {
        datasets: [lineDataset('μ₁ Samples', 'rgba(255, 99, 132, 0.7)')],
      },
      options: baseOptions('μ₁ Trace Plot', 'Step', 'Value'),
    }));

    /* μ₂ trace */
    initChart('component2-trace-chart', mu2TraceRef, () => ({
      type: 'line',
      data: {
        datasets: [lineDataset('μ₂ Samples', 'rgba(54, 162, 235, 0.7)')],
      },
      options: baseOptions('μ₂ Trace Plot', 'Step', 'Value'),
    }));
  }, [state]);

  /***** Update charts whenever state/parameters change *****/
  useEffect(() => {
    if (!state) return;

    // If state has been updated with new data and labels, ensure the chart is updated
    const { xMin, xMax } = xDomain();

    // Calculate empirical means for each component - moved outside chart blocks
    const c1 = state.data.filter((_, index) => state.labels[index] === 1);
    const c2 = state.data.filter((_, index) => state.labels[index] === 0);
    const empiricalMeans = {
      component1: calculateMean(c1),
      component2: calculateMean(c2),
    };

    /* -------- Distribution (target + densities + histogram) -------- */
    const distributionCtx = document.getElementById(
      'distribution-chart'
    ) as HTMLCanvasElement | null;
    if (!distributionCtx) return;

    // ---------------- Build datasets ----------------
    const localXVals = xVals;

    // Target density
    const targetDataset = lineDataset('Target', 'rgba(75, 192, 192, 1)');
    targetDataset.data = localXVals.map((x) => {
      const { mu1, mu2, sigma1, sigma2, p } = parameters;
      return {
        x,
        y: p * normalPdf(x, mu1, sigma1) + (1 - p) * normalPdf(x, mu2, sigma2),
      };
    });

    // Histogram
    const numBins = 30;
    const binWidth = (xMax - xMin) / numBins;
    const bins = Array(numBins).fill(0);
    for (const value of state.data ?? []) {
      const binIndex = Math.min(
        Math.floor((value - xMin) / binWidth),
        numBins - 1
      );
      if (binIndex >= 0) bins[binIndex]++;
    }
    const total = state.data?.length ?? 0;
    const histogramData = bins.map((count: number, i: number) => ({
      x: xMin + (i + 0.5) * binWidth,
      y: total ? count / (total * binWidth) : 0,
    }));

    // Posterior densities via KDE
    const mu1Dataset = lineDataset(
      'Posterior μ₁ Density',
      'rgba(255, 99, 132, 0.7)',
      true,
      'y2'
    );
    const mu2Dataset = lineDataset(
      'Posterior μ₂ Density',
      'rgba(54, 162, 235, 0.7)',
      true,
      'y2'
    );

    const mu1Samples = state.samples.mu1.slice(parameters.burnIn);
    const mu2Samples = state.samples.mu2.slice(parameters.burnIn);

    if (mu1Samples.length > 1) {
      mu1Dataset.data = localXVals.map((x) => ({
        x,
        y: kde(mu1Samples, x, 0.2),
      }));
    }

    if (mu2Samples.length > 1) {
      mu2Dataset.data = localXVals.map((x) => ({
        x,
        y: kde(mu2Samples, x, 0.2),
      }));
    }

    // ---------------- Annotation lines (empirical means) -------------
    const annotationConfig: AnnotationConfig = {};
    if (c1.length > 0) {
      annotationConfig.component1Mean = {
        type: 'line',
        xMin: empiricalMeans.component1,
        xMax: empiricalMeans.component1,
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        label: {
          display: true,
          content: `μ₁: ${empiricalMeans.component1.toFixed(2)}`,
          position: 'end', // show at top-right of line
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
          color: 'white',
          font: { size: 10, weight: 'normal' },
          padding: { x: 4, y: 2 },
          yAdjust: 3,
          xAdjust: 25,
        } as any,
      };
    }
    if (c2.length > 0) {
      annotationConfig.component2Mean = {
        type: 'line',
        xMin: empiricalMeans.component2,
        xMax: empiricalMeans.component2,
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        label: {
          display: true,
          content: `μ₂: ${empiricalMeans.component2.toFixed(2)}`,
          position: 'end',
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
          color: 'white',
          font: { size: 10, weight: 'normal' },
          padding: { x: 4, y: 2 },
          yAdjust: 3,
          xAdjust: 25,
        } as any,
      };
    }

    const datasets = [
      targetDataset,
      mu1Dataset,
      mu2Dataset,
      {
        label: 'Empirical Distribution',
        type: 'bar',
        backgroundColor: 'rgba(128, 128, 128, 0.3)',
        borderColor: 'rgba(128, 128, 128, 0.3)',
        borderWidth: 0,
        data: histogramData,
        yAxisID: 'y',
        barPercentage: 1,
        categoryPercentage: 1,
      },
    ];

    /* Create once, update thereafter */
    if (!distributionRef.current) {
      distributionRef.current = new Chart(distributionCtx, {
        type: 'line',
        data: { datasets: datasets as any },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: {
            padding: 0,
          },
          plugins: {
            title: { display: true, text: 'Distributions' },
            tooltip: { mode: 'index', intersect: false },
            annotation: { annotations: annotationConfig as any },
          },
          scales: {
            x: {
              type: 'linear',
              min: xMin,
              max: xMax,
              title: { display: true, text: 'x' },
              offset: false,
              grid: {
                offset: false,
              },
            },
            y: {
              title: { display: true, text: 'Probability Density' },
              position: 'left',
              grid: {
                offset: false,
              },
            },
            y2: {
              title: { display: true, text: 'Posterior Density' },
              position: 'right',
              grid: { drawOnChartArea: false },
              offset: false,
            },
          },
          animation: { duration: 0 },
        },
      });
    } else {
      // Update datasets and scales efficiently
      const chart = distributionRef.current;
      chart.data.datasets = datasets as any;
      chart.options.scales!.x = {
        ...(chart.options.scales!.x as any),
        min: xMin,
        max: xMax,
        offset: false,
        grid: {
          offset: false,
        },
      };
      chart.options.plugins!.annotation!.annotations = annotationConfig as any;
      chart.update('none');
    }

    // Compute a sensible centre (mean) and symmetric range (delta) for the y–axis.
    // When samples are available we use them; otherwise we fall back to the
    // corresponding prior parameters so that the axis still has reasonable ticks.
    const computeCenterAndDelta = (
      samples: number[],
      fallbackCenter: number,
      fallbackSigma: number
    ) => {
      if (samples.length === 0) {
        // No empirical samples yet – fall back to ±3σ around the prior mean.
        const delta = Math.max(3 * fallbackSigma, 1);
        return { center: fallbackCenter, delta } as const;
      }

      const center = calculateMean(samples);
      const min = Math.min(...samples);
      const max = Math.max(...samples);
      const delta =
        Math.max(Math.abs(min - center), Math.abs(max - center)) + 0.05;
      return { center, delta: delta * 1.01 } as const;
    };

    /* ---------- μ₁ Trace ---------- */
    if (mu1TraceRef.current) {
      const chart = mu1TraceRef.current;

      const { center, delta } = computeCenterAndDelta(
        state.samples.mu1,
        parameters.mu1,
        parameters.sigma1
      );

      chart.data.datasets[0].data = state.samples.mu1.map((y, x) => ({ x, y }));
      (chart.options.scales!.y as { min: number; max: number }).min =
        center - delta;
      (chart.options.scales!.y as { min: number; max: number }).max =
        center + delta;
      chart.options.plugins!.annotation!.annotations = {
        meanLine: {
          type: 'line',
          yMin: center,
          yMax: center,
          borderColor: 'rgba(128, 128, 128, 1)',
          borderWidth: 3,
          borderDash: [5, 5],
        },
      } as AnnotationConfig;
      chart.update();
    }

    /* ---------- μ₂ Trace ---------- */
    if (mu2TraceRef.current) {
      const chart = mu2TraceRef.current;

      const { center, delta } = computeCenterAndDelta(
        state.samples.mu2,
        parameters.mu2,
        parameters.sigma2
      );

      chart.data.datasets[0].data = state.samples.mu2.map((y, x) => ({ x, y }));
      (chart.options.scales!.y as { min: number; max: number }).min =
        center - delta;
      (chart.options.scales!.y as { min: number; max: number }).max =
        center + delta;
      chart.options.plugins!.annotation!.annotations = {
        meanLine: {
          type: 'line',
          yMin: center,
          yMax: center,
          borderColor: 'rgba(128, 128, 128, 1)',
          borderWidth: 3,
          borderDash: [5, 5],
        },
      } as AnnotationConfig;
      chart.update();
    }
  }, [state, parameters, xVals, xDomain]);

  /***** Render *****/
  return (
    <div className="w-full space-y-8">
      {/* Distribution plot + Statistics side-by-side */}
      <div className="grid w-full grid-cols-4 gap-8">
        {/* 25% width for statistics */}
        <div className="col-span-1 w-full">
          <Statistics state={state} />
        </div>
        {/* 75% width for distribution plot */}
        <div className="col-span-3 w-full rounded-lg bg-white p-4 shadow">
          <canvas id="distribution-chart" className="h-[400px] w-full" />
        </div>
      </div>

      {/* Trace plots */}
      <div className="grid w-full grid-cols-1 gap-8 lg:grid-cols-2">
        <div className="w-full rounded-lg bg-white p-4 shadow">
          <canvas id="component1-trace-chart" className="h-[400px] w-full" />
        </div>
        <div className="w-full rounded-lg bg-white p-4 shadow">
          <canvas id="component2-trace-chart" className="h-[400px] w-full" />
        </div>
      </div>
    </div>
  );
}
