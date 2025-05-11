import { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import annotationPlugin, { AnnotationOptions, AnnotationTypeRegistry } from 'chartjs-plugin-annotation';

Chart.register(...registerables, annotationPlugin);

type AnnotationConfig = {
  [key: string]: AnnotationOptions<keyof AnnotationTypeRegistry>;
};

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
  mu1: number;
  mu2: number;
  acceptance_ratio: number;
  samples: {
    mu1: number[];
    mu2: number[];
  };
  steps: Array<{ mu1: number; mu2: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
}

interface ChartsProps {
  state: SimulationState | null;
  parameters: Parameters;
}

// Helper for kernel density estimation
function kernelDensityEstimate(samples: number[], xVals: number[], bandwidth = 0.2) {
  const norm = 1 / (Math.sqrt(2 * Math.PI) * bandwidth * samples.length);
  return xVals.map(x => ({
    x,
    y: samples.reduce((sum, xi) => sum + Math.exp(-0.5 * Math.pow((x - xi) / bandwidth, 2)), 0) * norm
  }));
}

export default function Charts({ state, parameters }: ChartsProps) {
  const distributionChartRef = useRef<Chart | null>(null);
  const traceChartRef = useRef<Chart | null>(null);
  const component1TraceChartRef = useRef<Chart | null>(null);
  const component2TraceChartRef = useRef<Chart | null>(null);
  const mu1DensityChartRef = useRef<Chart | null>(null);
  const mu2DensityChartRef = useRef<Chart | null>(null);

  // Initialize charts
  useEffect(() => {
    if (!state) return;

    // Distribution chart
    const distributionCtx = document.getElementById('distribution-chart') as HTMLCanvasElement;
    if (distributionCtx && !distributionChartRef.current) {
      distributionChartRef.current = new Chart(distributionCtx, {
        type: 'line',
        data: {
          datasets: [
            {
              label: 'Target GMM',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              data: [],
              yAxisID: 'y',
            },
            {
              label: 'Posterior μ₁ Density',
              borderColor: 'rgba(255, 99, 132, 0.7)',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              borderDash: [2, 2],
              data: [],
              yAxisID: 'y2',
            },
            {
              label: 'Posterior μ₂ Density',
              borderColor: 'rgba(54, 162, 235, 0.7)',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              borderDash: [2, 2],
              data: [],
              yAxisID: 'y2',
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'GMM Distribution vs. Samples'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            },
            annotation: {
              annotations: {}
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'x'
              },
              min: -4,
              max: 4
            },
            y: {
              title: {
                display: true,
                text: 'Probability Density'
              },
              position: 'left',
            },
            y2: {
              title: {
                display: true,
                text: 'Posterior Density'
              },
              position: 'right',
              grid: {
                drawOnChartArea: false
              }
            }
          },
          animation: false
        }
      });
    }

    // Trace chart
    const traceCtx = document.getElementById('trace-chart') as HTMLCanvasElement;
    if (traceCtx && !traceChartRef.current) {
      traceChartRef.current = new Chart(traceCtx, {
        type: 'line',
        data: {
          datasets: [
            {
              label: 'μ₁ Samples',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              data: []
            },
            {
              label: 'μ₂ Samples',
              borderColor: 'rgba(255, 99, 132, 1)',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              data: []
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'Full Trace Plot'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            },
            annotation: {
              annotations: {}
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'Step'
              },
              min: 0
            },
            y: {
              title: {
                display: true,
                text: 'Value'
              }
            }
          },
          animation: false
        }
      });
    }

    // Component 1 trace chart
    const component1TraceCtx = document.getElementById('component1-trace-chart') as HTMLCanvasElement;
    if (component1TraceCtx && !component1TraceChartRef.current) {
      component1TraceChartRef.current = new Chart(component1TraceCtx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'μ₁ Samples',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            data: []
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'μ₁ Trace Plot'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            },
            annotation: {
              annotations: {}
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'Step'
              },
              min: 0
            },
            y: {
              title: {
                display: true,
                text: 'Value'
              },
              min: parameters.mean1 - 0.5,
              max: parameters.mean1 + 0.5
            }
          },
          animation: false
        }
      });
    }

    // Component 2 trace chart
    const component2TraceCtx = document.getElementById('component2-trace-chart') as HTMLCanvasElement;
    if (component2TraceCtx && !component2TraceChartRef.current) {
      component2TraceChartRef.current = new Chart(component2TraceCtx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'μ₂ Samples',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            data: []
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'μ₂ Trace Plot'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            },
            annotation: {
              annotations: {}
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'Step'
              },
              min: 0
            },
            y: {
              title: {
                display: true,
                text: 'Value'
              },
              min: parameters.mean2 - 0.5,
              max: parameters.mean2 + 0.5
            }
          },
          animation: false
        }
      });
    }

    // μ₁ density chart
    const mu1DensityCtx = document.getElementById('mu1-density-chart') as HTMLCanvasElement;
    if (mu1DensityCtx && !mu1DensityChartRef.current) {
      mu1DensityChartRef.current = new Chart(mu1DensityCtx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'Posterior Density μ₁',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            data: []
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'Posterior Density of μ₁'
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'μ₁'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Density'
              }
            }
          },
          animation: false
        }
      });
    }

    // μ₂ density chart
    const mu2DensityCtx = document.getElementById('mu2-density-chart') as HTMLCanvasElement;
    if (mu2DensityCtx && !mu2DensityChartRef.current) {
      mu2DensityChartRef.current = new Chart(mu2DensityCtx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'Posterior Density μ₂',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            data: []
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: 'Posterior Density of μ₂'
            }
          },
          scales: {
            x: {
              type: 'linear',
              title: {
                display: true,
                text: 'μ₂'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Density'
              }
            }
          },
          animation: false
        }
      });
    }
  }, [state]);

  // Update charts when state or parameters change
  useEffect(() => {
    if (!state) return;

    // Update distribution chart
    if (distributionChartRef.current) {
      // Use a fixed x grid for all curves
      const xVals = Array.from({ length: 100 }, (_, i) => -4 + (8 * i) / 99);
      
      // Update x-axis scale based on means
      const xMin = Math.min(parameters.mean1, parameters.mean2) - 2;
      const xMax = Math.max(parameters.mean1, parameters.mean2) + 2;
      if (distributionChartRef.current?.options?.scales?.x) {
        distributionChartRef.current.options.scales.x.min = xMin;
        distributionChartRef.current.options.scales.x.max = xMax;
      }

      // Target GMM (fixed)
      distributionChartRef.current.data.datasets[0].data = xVals.map(x => {
        const p1 = parameters.mixtureWeight * (1 / Math.sqrt(2 * Math.PI * parameters.variance1)) * Math.exp(-0.5 * Math.pow((x - parameters.mean1) / Math.sqrt(parameters.variance1), 2));
        const p2 = (1 - parameters.mixtureWeight) * (1 / Math.sqrt(2 * Math.PI * parameters.variance2)) * Math.exp(-0.5 * Math.pow((x - parameters.mean2) / Math.sqrt(parameters.variance2), 2));
        return { x, y: p1 + p2 };
      });

      // Posterior μ₁ density (kernel density estimate)
      const mu1Samples = state.samples.mu1.slice(parameters.burnIn);
      if (mu1Samples.length > 1) {
        const density = kernelDensityEstimate(mu1Samples, xVals);
        distributionChartRef.current.data.datasets[1].data = density;
      } else {
        distributionChartRef.current.data.datasets[1].data = [];
      }

      // Posterior μ₂ density (kernel density estimate)
      const mu2Samples = state.samples.mu2.slice(parameters.burnIn);
      if (mu2Samples.length > 1) {
        const density = kernelDensityEstimate(mu2Samples, xVals);
        distributionChartRef.current.data.datasets[2].data = density;
      } else {
        distributionChartRef.current.data.datasets[2].data = [];
      }

      // Update annotations
      if (distributionChartRef.current?.options?.plugins?.annotation) {
        distributionChartRef.current.options.plugins.annotation.annotations = {};
      }
      distributionChartRef.current.update();
    }

    // Update trace charts
    if (traceChartRef.current) {
      const mu1Samples = state.samples.mu1
        .map((value, index) => ({ x: index, y: value }))
        .filter(point => point.x >= parameters.burnIn);
      const mu2Samples = state.samples.mu2
        .map((value, index) => ({ x: index, y: value }))
        .filter(point => point.x >= parameters.burnIn);
      
      // Update y-axis scale based on means
      const yMin = Math.min(parameters.mean1, parameters.mean2) - 1;
      const yMax = Math.max(parameters.mean1, parameters.mean2) + 1;
      if (traceChartRef.current?.options?.scales?.y) {
        traceChartRef.current.options.scales.y.min = yMin;
        traceChartRef.current.options.scales.y.max = yMax;
      }

      traceChartRef.current.data.datasets[0].data = mu1Samples;
      traceChartRef.current.data.datasets[1].data = mu2Samples;
      
      if (traceChartRef.current?.options?.plugins?.annotation) {
        traceChartRef.current.options.plugins.annotation.annotations = {
          burnIn: {
            type: 'box',
            xMin: 0,
            xMax: parameters.burnIn,
            backgroundColor: 'rgba(200,200,200,0.15)',
            borderWidth: 0,
            label: {
              content: 'Burn-in',
              position: 'start',
              color: '#888'
            }
          },
          mean1Line: {
            type: 'line',
            yMin: parameters.mean1,
            yMax: parameters.mean1,
            borderColor: 'green',
            borderWidth: 2,
            borderDash: [5, 5]
          },
          mean2Line: {
            type: 'line',
            yMin: parameters.mean2,
            yMax: parameters.mean2,
            borderColor: 'orange',
            borderWidth: 2,
            borderDash: [5, 5]
          }
        } as AnnotationConfig;
      }
      traceChartRef.current.update();
    }

    // Update component trace charts
    if (component1TraceChartRef.current) {
      component1TraceChartRef.current.data.datasets[0].data = state.samples.mu1
        .slice(parameters.burnIn, parameters.numSamples)
        .map((value, index) => ({ x: parameters.burnIn + index, y: value }));

      // Update y-axis scale for component 1
      if (component1TraceChartRef.current?.options?.scales?.y) {
        component1TraceChartRef.current.options.scales.y.min = parameters.mean1 - 1;
        component1TraceChartRef.current.options.scales.y.max = parameters.mean1 + 1;
      }

      if (component1TraceChartRef.current.options?.plugins?.annotation) {
        component1TraceChartRef.current.options.plugins.annotation.annotations = {
          burnIn: {
            type: 'box',
            xMin: 0,
            xMax: parameters.burnIn,
            backgroundColor: 'rgba(200,200,200,0.15)',
            borderWidth: 0
          },
          meanLine: {
            type: 'line',
            yMin: parameters.mean1,
            yMax: parameters.mean1,
            borderColor: 'green',
            borderWidth: 2,
            borderDash: [5, 5]
          }
        } as AnnotationConfig;
      }
      component1TraceChartRef.current.update();
    }

    if (component2TraceChartRef.current) {
      component2TraceChartRef.current.data.datasets[0].data = state.samples.mu2
        .slice(parameters.burnIn, parameters.numSamples)
        .map((value, index) => ({ x: parameters.burnIn + index, y: value }));

      // Update y-axis scale for component 2
      if (component2TraceChartRef.current?.options?.scales?.y) {
        component2TraceChartRef.current.options.scales.y.min = parameters.mean2 - 1;
        component2TraceChartRef.current.options.scales.y.max = parameters.mean2 + 1;
      }

      if (component2TraceChartRef.current.options?.plugins?.annotation) {
        component2TraceChartRef.current.options.plugins.annotation.annotations = {
          burnIn: {
            type: 'box',
            xMin: 0,
            xMax: parameters.burnIn,
            backgroundColor: 'rgba(200,200,200,0.15)',
            borderWidth: 0
          },
          meanLine: {
            type: 'line',
            yMin: parameters.mean2,
            yMax: parameters.mean2,
            borderColor: 'orange',
            borderWidth: 2,
            borderDash: [5, 5]
          }
        } as AnnotationConfig;
      }
      component2TraceChartRef.current.update();
    }
  }, [state, parameters]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      <div className="bg-white p-4 rounded-lg shadow">
        <canvas id="distribution-chart" className="h-[350px]"></canvas>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <canvas id="trace-chart" className="h-[350px]"></canvas>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <canvas id="component1-trace-chart" className="h-[350px]"></canvas>
      </div>
      <div className="bg-white p-4 rounded-lg shadow">
        <canvas id="component2-trace-chart" className="h-[350px]"></canvas>
      </div>
    </div>
  );
} 