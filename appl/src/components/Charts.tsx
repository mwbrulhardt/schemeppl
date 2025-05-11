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
  current_x: number;
  proposed_x: number;
  acceptance_ratio: number;
  samples: number[];
  steps: Array<{ x: number; accepted: boolean }>;
  distribution: Array<{ x: number; pdf: number }>;
  histogram: Array<{ x: number; frequency: number }>;
}

interface ChartsProps {
  state: SimulationState | null;
  parameters: Parameters;
}

export default function Charts({ state, parameters }: ChartsProps) {
  const distributionChartRef = useRef<Chart | null>(null);
  const traceChartRef = useRef<Chart | null>(null);
  const component1TraceChartRef = useRef<Chart | null>(null);
  const component2TraceChartRef = useRef<Chart | null>(null);

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
              data: []
            },
            {
              label: 'Sample Histogram',
              borderColor: 'rgba(153, 102, 255, 1)',
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
              }
            },
            y: {
              title: {
                display: true,
                text: 'Probability Density'
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
              label: 'All Samples',
              borderColor: 'rgba(75, 192, 192, 1)',
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
            label: 'Component 1 Samples',
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
              text: 'Component 1 Trace Plot'
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

    // Component 2 trace chart
    const component2TraceCtx = document.getElementById('component2-trace-chart') as HTMLCanvasElement;
    if (component2TraceCtx && !component2TraceChartRef.current) {
      component2TraceChartRef.current = new Chart(component2TraceCtx, {
        type: 'line',
        data: {
          datasets: [{
            label: 'Component 2 Samples',
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
              text: 'Component 2 Trace Plot'
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
  }, [state]);

  // Update charts when state changes
  useEffect(() => {
    if (!state) return;

    // Update distribution chart
    if (distributionChartRef.current) {
      distributionChartRef.current.data.datasets[0].data = state.distribution.map(point => ({
        x: point.x,
        y: point.pdf
      }));
      distributionChartRef.current.data.datasets[1].data = state.histogram.map(bin => ({
        x: bin.x,
        y: bin.frequency
      }));
      if (distributionChartRef.current?.options?.plugins?.annotation) {
        distributionChartRef.current.options.plugins.annotation.annotations = {
          currentLine: {
            type: 'line',
            xMin: state.current_x,
            xMax: state.current_x,
            borderColor: 'blue',
            borderWidth: 2
          },
          proposedLine: {
            type: 'line',
            xMin: state.proposed_x,
            xMax: state.proposed_x,
            borderColor: 'red',
            borderWidth: 2,
            borderDash: [5, 5]
          }
        } as AnnotationConfig;
      }
      distributionChartRef.current.update();
    }

    // Update trace charts
    if (traceChartRef.current) {
      const allSamples = state.samples.map((value, index) => ({ x: index, y: value }));
      traceChartRef.current.data.datasets[0].data = allSamples;
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
      traceChartRef.current?.update();
    }

    // Update component trace charts
    const midpoint = (parameters.mean1 + parameters.mean2) / 2;
    const comp1Samples = state.samples.filter(x => x < midpoint);
    const comp2Samples = state.samples.filter(x => x >= midpoint);

    if (component1TraceChartRef.current) {
      component1TraceChartRef.current.data.datasets[0].data = comp1Samples.map((value, index) => ({
        x: index,
        y: value
      }));
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
      component2TraceChartRef.current.data.datasets[0].data = comp2Samples.map((value, index) => ({
        x: index,
        y: value
      }));
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