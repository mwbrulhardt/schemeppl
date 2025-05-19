import { Parameters } from '@/types/simulation';
import { SimulationState } from '@/hooks/useSimulator';
import { useCallback, useEffect, useState, useMemo } from 'react';
import 'katex/dist/katex.min.css';
import katex from 'katex';

// Parameter constraints
const PARAMETER_CONSTRAINTS = {
  mu1: { min: -Infinity, max: Infinity },
  mu2: { min: -Infinity, max: Infinity },
  sigma1: { min: 0.1, max: Infinity },
  sigma2: { min: 0.1, max: Infinity },
  p: { min: 0.01, max: 0.99 },
  proposalStdDev1: { min: 0.1, max: Infinity },
  proposalStdDev2: { min: 0.1, max: Infinity },
  numSteps: { min: 100, max: 10000 },
  burnIn: { min: 0, max: 2000 },
  delay: { min: 0, max: 1000 },
  seed: { min: 0, max: Infinity },
  sampleSize: { min: 10, max: 1000 },
} as const;

interface ControlsProps {
  parameters: Parameters;
  state: SimulationState | null;
  onUpdateParameters: (params: Partial<Parameters>) => void;
  isRunning: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
}

export default function Controls({
  parameters,
  state,
  onUpdateParameters,
  isRunning,
  onStart,
  onPause,
  onReset,
}: ControlsProps) {
  // Local state for input values and validity
  const [inputValues, setInputValues] = useState<
    Record<keyof Parameters, string>
  >({
    mu1: parameters.mu1.toString(),
    mu2: parameters.mu2.toString(),
    sigma1: parameters.sigma1.toString(),
    sigma2: parameters.sigma2.toString(),
    p: parameters.p.toString(),
    proposalStdDev1: parameters.proposalStdDev1.toString(),
    proposalStdDev2: parameters.proposalStdDev1.toString(),
    numSteps: parameters.numSteps.toString(),
    burnIn: parameters.burnIn.toString(),
    delay: parameters.delay.toString(),
    seed: parameters.seed.toString(),
    sampleSize: parameters.sampleSize.toString(),
  });
  const [inputValidity, setInputValidity] = useState<
    Record<keyof Parameters, boolean>
  >({
    mu1: true,
    mu2: true,
    sigma1: true,
    sigma2: true,
    p: true,
    proposalStdDev1: true,
    proposalStdDev2: true,
    numSteps: true,
    burnIn: true,
    delay: true,
    seed: true,
    sampleSize: true,
  });

  // Validation rules for each parameter
  const validators = useMemo(
    () => ({
      mu1: (v: string) => !isNaN(Number(v)),
      mu2: (v: string) => !isNaN(Number(v)),
      sigma1: (v: string) =>
        !isNaN(Number(v)) && Number(v) >= PARAMETER_CONSTRAINTS.sigma1.min,
      sigma2: (v: string) =>
        !isNaN(Number(v)) && Number(v) >= PARAMETER_CONSTRAINTS.sigma2.min,
      p: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.p.min &&
        Number(v) <= PARAMETER_CONSTRAINTS.p.max,
      proposalStdDev1: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.proposalStdDev1.min,
      proposalStdDev2: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.proposalStdDev2.min,
      numSteps: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.numSteps.min &&
        Number(v) <= PARAMETER_CONSTRAINTS.numSteps.max,
      burnIn: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.burnIn.min &&
        Number(v) <= PARAMETER_CONSTRAINTS.burnIn.max,
      delay: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.delay.min &&
        Number(v) <= PARAMETER_CONSTRAINTS.delay.max,
      seed: (v: string) =>
        !isNaN(Number(v)) && Number(v) >= PARAMETER_CONSTRAINTS.seed.min,
      sampleSize: (v: string) =>
        !isNaN(Number(v)) &&
        Number(v) >= PARAMETER_CONSTRAINTS.sampleSize.min &&
        Number(v) <= PARAMETER_CONSTRAINTS.sampleSize.max,
    }),
    []
  );

  // Update local state when parameters change
  useEffect(() => {
    setInputValues({
      mu1: parameters.mu1.toString(),
      mu2: parameters.mu2.toString(),
      sigma1: parameters.sigma1.toString(),
      sigma2: parameters.sigma2.toString(),
      p: parameters.p.toString(),
      proposalStdDev1: parameters.proposalStdDev1.toString(),
      proposalStdDev2: parameters.proposalStdDev2.toString(),
      numSteps: parameters.numSteps.toString(),
      burnIn: parameters.burnIn.toString(),
      delay: parameters.delay.toString(),
      seed: parameters.seed.toString(),
      sampleSize: parameters.sampleSize.toString(),
    });
    setInputValidity({
      mu1: true,
      mu2: true,
      sigma1: true,
      sigma2: true,
      p: true,
      proposalStdDev1: true,
      proposalStdDev2: true,
      numSteps: true,
      burnIn: true,
      delay: true,
      seed: true,
      sampleSize: true,
    });
  }, [parameters]);

  /**
   * Update a single parameter while the user is typing.
   *
   * Rules:
   *   • Local input state (`inputValues`) is always updated – user can type freely.
   *   • Each field's `inputValidity` is updated so that invalid values are highlighted.
   *   • The expensive part (propagating the change to the simulator/worker) only
   *     happens when the *entire* constraint set is satisfied – in particular
   *     we require `mu1 < mu2` before calling `onUpdateParameters` for either
   *     of those two fields.
   */
  const handleParameterChange = useCallback(
    (key: keyof Parameters, value: string) => {
      // Immediately reflect the change in the local form state
      setInputValues((prev) => ({ ...prev, [key]: value }));

      // Build helper snapshot that includes the just-modified field
      const nextValues = { ...inputValues, [key]: value };

      // Check if the input is a partial number (like "-" or "-0" or ".")
      const isPartialNumber = /^-?\.?$|^-?0?\.?$/.test(value) && value !== '0';

      // Per-field basic validation (number ranges, etc.)
      // Only validate if it's not a partial number
      const basicValid = isPartialNumber || validators[key](value);

      // Special cross-field constraint for the means
      const mu1Num = parseFloat(nextValues.mu1);
      const mu2Num = parseFloat(nextValues.mu2);
      const muOrderingSatisfied =
        !isNaN(mu1Num) && !isNaN(mu2Num) ? mu1Num < mu2Num : true;

      // Update validity map – mu fields depend on each other
      setInputValidity((prev) => {
        const updated: typeof prev = { ...prev };
        if (key === 'mu1' || key === 'mu2') {
          updated.mu1 =
            (isPartialNumber || validators.mu1(nextValues.mu1)) &&
            muOrderingSatisfied;
          updated.mu2 =
            (isPartialNumber || validators.mu2(nextValues.mu2)) &&
            muOrderingSatisfied;
        } else {
          updated[key] = basicValid;
        }
        return updated;
      });

      // Only forward to the simulator when we have a complete, valid number
      const readyToPropagate =
        basicValid &&
        (key === 'mu1' || key === 'mu2' ? muOrderingSatisfied : true) &&
        !isPartialNumber &&
        !isNaN(parseFloat(value));

      if (readyToPropagate) {
        onUpdateParameters({ [key]: parseFloat(value) });
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onUpdateParameters, validators, inputValues]
  );

  // Add a separate handler for the delay slider
  const handleDelayChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setInputValues((prev) => ({ ...prev, delay: value }));

      // The delay value is always valid based on the slider constraints
      const numericValue = parseFloat(value);

      // Update the parameter immediately
      onUpdateParameters({ delay: numericValue });
    },
    [onUpdateParameters]
  );

  // Helper to get input class
  const getInputClass = (key: keyof Parameters) =>
    `w-full p-2 border rounded ${
      inputValidity[key] ? '' : 'border-red-500 ring-2 ring-red-200'
    } ${isRunning ? 'opacity-50 bg-gray-100 cursor-not-allowed' : ''}`;

  // Compute progress bar values (based on simulation state)
  const { progressPercent, postBurnSteps } = useMemo(() => {
    if (!state) return { progressPercent: 0, postBurnSteps: 0 };
    const postBurn = Math.max(state.steps.length - parameters.burnIn, 0);
    const prog = Math.min(postBurn / parameters.numSteps, 1);
    return { progressPercent: prog * 100, postBurnSteps: postBurn };
  }, [state, parameters.burnIn, parameters.numSteps]);

  // Determine if the simulation has completed all requested samples
  const simulationCompleted =
    !isRunning && postBurnSteps >= parameters.numSteps;

  // Main control button click behaviour
  const handleMainButtonClick = () => {
    if (isRunning) {
      onPause();
    } else if (simulationCompleted) {
      // Restart implies a fresh run
      onReset();
      onStart();
    } else {
      onStart();
    }
  };

  // Helper to render KaTeX label
  const renderKaTeXLabel = (latex: string) => {
    return (
      <div
        className="w-16"
        dangerouslySetInnerHTML={{ __html: katex.renderToString(latex) }}
      />
    );
  };

  return (
    <div className="bg-gray-50 p-4 rounded-lg mb-8">
      <h2 className="text-xl font-semibold mb-4">Parameters</h2>
      {/* Mixture Model Section */}
      <div className="space-y-6">
        <div
          className={`bg-white p-4 rounded-lg shadow ${isRunning ? 'opacity-75' : ''}`}
        >
          <h3
            className={`text-lg font-medium mb-4 ${isRunning ? 'text-gray-500' : ''}`}
          >
            Mixture Model
          </h3>

          {/* Components Grid */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {/* Component 1 */}
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">
                Component 1
              </h4>
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-3">
                    {renderKaTeXLabel('\\mu_1')}
                    <input
                      type="number"
                      value={inputValues.mu1}
                      onChange={(e) =>
                        handleParameterChange('mu1', e.target.value)
                      }
                      step="0.1"
                      className={getInputClass('mu1')}
                      disabled={isRunning}
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    {renderKaTeXLabel('\\sigma_1')}
                    <input
                      type="number"
                      value={inputValues.sigma1}
                      onChange={(e) =>
                        handleParameterChange('sigma1', e.target.value)
                      }
                      min={PARAMETER_CONSTRAINTS.sigma1.min}
                      step="0.1"
                      className={getInputClass('sigma1')}
                      disabled={isRunning}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {renderKaTeXLabel('\\tau_1')}
                  <input
                    type="number"
                    value={inputValues.proposalStdDev1}
                    onChange={(e) =>
                      handleParameterChange('proposalStdDev1', e.target.value)
                    }
                    min={PARAMETER_CONSTRAINTS.proposalStdDev1.min}
                    step="0.1"
                    className={getInputClass('proposalStdDev1')}
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>

            {/* Component 2 */}
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">
                Component 2
              </h4>
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-3">
                    {renderKaTeXLabel('\\mu_2')}
                    <input
                      type="number"
                      value={inputValues.mu2}
                      onChange={(e) =>
                        handleParameterChange('mu2', e.target.value)
                      }
                      step="0.1"
                      className={getInputClass('mu2')}
                      disabled={isRunning}
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    {renderKaTeXLabel('\\sigma_2')}
                    <input
                      type="number"
                      value={inputValues.sigma2}
                      onChange={(e) =>
                        handleParameterChange('sigma2', e.target.value)
                      }
                      min={PARAMETER_CONSTRAINTS.sigma2.min}
                      step="0.1"
                      className={getInputClass('sigma2')}
                      disabled={isRunning}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {renderKaTeXLabel('\\tau_2')}
                  <input
                    type="number"
                    value={inputValues.proposalStdDev2}
                    onChange={(e) =>
                      handleParameterChange('proposalStdDev2', e.target.value)
                    }
                    min={PARAMETER_CONSTRAINTS.proposalStdDev2.min}
                    step="0.1"
                    className={getInputClass('proposalStdDev2')}
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Mixture Proportion (π) and Sample Size sliders */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* π Slider */}
            <div>
              <div className="flex items-center justify-between mb-2">
                {renderKaTeXLabel('\\pi')}
                <span className="text-sm text-gray-600">{inputValues.p}</span>
              </div>
              <input
                type="range"
                value={inputValues.p}
                onChange={(e) => handleParameterChange('p', e.target.value)}
                min={PARAMETER_CONSTRAINTS.p.min}
                max={PARAMETER_CONSTRAINTS.p.max}
                step="0.01"
                className={`${getInputClass('p')} ${isRunning ? 'bg-green-50' : ''}`}
                disabled={isRunning}
              />
              {!inputValidity.p && (
                <div className="text-xs text-red-600 mt-1">
                  Enter a value between 0 and 1 (exclusive).
                </div>
              )}
            </div>

            {/* Sample Size Slider */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium">Sample Size</label>
                <span className="text-sm text-gray-600">
                  {inputValues.sampleSize}
                </span>
              </div>
              <input
                type="range"
                value={inputValues.sampleSize}
                onChange={(e) =>
                  handleParameterChange('sampleSize', e.target.value)
                }
                min={PARAMETER_CONSTRAINTS.sampleSize.min}
                max={PARAMETER_CONSTRAINTS.sampleSize.max}
                step="10"
                className={`${getInputClass('sampleSize')} ${isRunning ? 'bg-green-50' : ''}`}
                disabled={isRunning}
              />
            </div>
          </div>
        </div>

        {/* Simulation Section */}
        <div className="flex flex-col md:flex-row gap-6">
          {/* Simulation Parameters */}
          <div
            className={`bg-white p-4 rounded-lg shadow flex-1 ${isRunning ? 'opacity-75' : ''}`}
          >
            <h3
              className={`text-lg font-medium mb-3 ${isRunning ? 'text-gray-500' : ''}`}
            >
              Simulation
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <label className="w-32 text-sm font-medium">
                  Number of Steps
                </label>
                <input
                  type="number"
                  value={inputValues.numSteps}
                  onChange={(e) =>
                    handleParameterChange('numSteps', e.target.value)
                  }
                  min={PARAMETER_CONSTRAINTS.numSteps.min}
                  max={PARAMETER_CONSTRAINTS.numSteps.max}
                  step="100"
                  className={getInputClass('numSteps')}
                  disabled={isRunning}
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="w-32 text-sm font-medium">
                  Burn-In Period
                </label>
                <input
                  type="number"
                  value={inputValues.burnIn}
                  onChange={(e) =>
                    handleParameterChange('burnIn', e.target.value)
                  }
                  min={PARAMETER_CONSTRAINTS.burnIn.min}
                  max={PARAMETER_CONSTRAINTS.burnIn.max}
                  step="10"
                  className={getInputClass('burnIn')}
                  disabled={isRunning}
                />
              </div>
            </div>
          </div>

          {/* Simulation Controls */}
          <div className="bg-white p-4 rounded-lg shadow w-full md:w-72 flex flex-col gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium">Animation Speed</label>
                <span className="text-sm text-gray-600">
                  {parameters.delay}ms
                </span>
              </div>
              <input
                type="range"
                value={inputValues.delay}
                onChange={handleDelayChange}
                min={PARAMETER_CONSTRAINTS.delay.min}
                max={PARAMETER_CONSTRAINTS.delay.max}
                step="10"
                className={`${getInputClass('delay')} ${isRunning ? 'bg-green-50' : ''}`}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Fast</span>
                <span>Slow</span>
              </div>
            </div>
            <div className="flex gap-4 mt-2">
              {/* Determine the label: Pause / Restart / Continue / Start */}
              <button
                onClick={handleMainButtonClick}
                className={`flex-1 px-4 py-2 rounded text-white ${
                  isRunning
                    ? 'bg-red-500 hover:bg-red-600'
                    : 'bg-green-500 hover:bg-green-600'
                }`}
              >
                {isRunning
                  ? 'Pause'
                  : simulationCompleted
                    ? 'Restart'
                    : state && state.steps && state.steps.length > 0
                      ? 'Continue'
                      : 'Start'}
              </button>
              <button
                onClick={onReset}
                className="flex-1 px-4 py-2 rounded bg-gray-200 hover:bg-gray-300"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-600 mb-2">Progress</h3>
        <div className="w-full bg-gray-200 rounded h-2 overflow-hidden">
          <div
            className="bg-blue-500 h-full"
            style={{ width: `${progressPercent.toFixed(1)}%` }}
          />
        </div>
        <p className="text-xs text-gray-500 mt-1 text-right">
          {postBurnSteps}/{parameters.numSteps} samples
        </p>
      </div>
    </div>
  );
}
