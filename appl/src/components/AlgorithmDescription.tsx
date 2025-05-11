export default function AlgorithmDescription() {
  return (
    <div className="bg-white p-6 rounded-lg shadow mb-8">
      <h2 className="text-xl font-semibold mb-4">Algorithm Description</h2>
      <p className="mb-4">
        The Metropolis-Hastings algorithm is used to sample from a Gaussian Mixture Model (GMM) with two components. 
        The target distribution is:
      </p>
      <p className="mb-4 font-mono text-center">
        p(x) = π * N(x|μ₁,σ₁²) + (1-π) * N(x|μ₂,σ₂²)
      </p>
      <p className="mb-4">where:</p>
      <ul className="list-disc pl-6 mb-4">
        <li>π is the mixture weight (probability of component 1)</li>
        <li>μ₁, μ₂ are the means of the two components</li>
        <li>σ₁², σ₂² are the variances of the two components</li>
      </ul>
      <p className="mb-4">The algorithm works as follows:</p>
      <ol className="list-decimal pl-6">
        <li>Start with an initial point x₀.</li>
        <li>At each step t, propose a new point x* from a proposal distribution q(x*|xₜ).</li>
        <li>Calculate the acceptance probability α = min(1, p(x*)/p(xₜ)) where p is the GMM density.</li>
        <li>Generate u ~ Uniform(0,1). If u &lt; α, accept the proposal and set xₜ₊₁ = x*. Otherwise, set xₜ₊₁ = xₜ.</li>
        <li>Repeat steps 2-4 to generate a sequence of samples.</li>
      </ol>
    </div>
  );
} 