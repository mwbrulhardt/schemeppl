// Calculate mean from samples
export const calculateMean = (samples: number[]) => {
  if (samples.length === 0) return 0;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
};

// Calculate standard deviations from samples
export const calculateStdDev = (samples: number[]) => {
  if (samples.length === 0) return 0;
  const mean = calculateMean(samples);
  const variance =
    samples.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
    (samples.length - 1);
  return Math.sqrt(variance);
};

// Compute the kernel density estimate
export const kde = (samples: number[], x: number, bandwidth: number) => {
  const norm = 1 / (Math.sqrt(2 * Math.PI) * bandwidth * samples.length);
  return (
    samples.reduce(
      (sum, xi) => sum + Math.exp(-0.5 * Math.pow((x - xi) / bandwidth, 2)),
      0
    ) * norm
  );
};

// Normal PDF
export function normalPdf(x: number, mu: number, sigma: number) {
  return (
    (1 / (sigma * Math.sqrt(2 * Math.PI))) *
    Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2))
  );
}
