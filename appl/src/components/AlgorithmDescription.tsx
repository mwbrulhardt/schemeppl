import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

export default function AlgorithmDescription() {
  return (
    <div className="bg-white p-6 rounded-lg shadow mb-8">
      <h2 className="text-xl font-semibold mb-4">Algorithm Description</h2>
      <p className="mb-4">
        The Metropolis-Hastings algorithm is used to sample from a Gaussian Mixture Model (GMM) with two components. 
        The target distribution is:
      </p>
      <div className="mb-4 text-center">
        <BlockMath math={
          'p(x) = \\pi \\cdot N(x|\\mu_1,\\sigma_1^2) + (1-\\pi) \\cdot N(x|\\mu_2,\\sigma_2^2)'
        } />
      </div>
      <p className="mb-4">where:</p>
      <ul className="list-disc pl-6 mb-4">
        <li><InlineMath math={'\\pi'} /> is the mixture weight (probability of component 1)</li>
        <li><InlineMath math={'\\mu_1, \\mu_2'} /> are the means of the two components</li>
        <li><InlineMath math={'\\sigma_1^2, \\sigma_2^2'} /> are the variances of the two components</li>
      </ul>
      <p className="mb-4">The algorithm works as follows:</p>
      <ol className="list-decimal pl-6">
        <li>Start with an initial point <InlineMath math={'x_0'} />.</li>
        <li>At each step <InlineMath math={'t'} />, propose a new point <InlineMath math={'x^*'} /> from a proposal distribution <InlineMath math={'q(x^*|x_t)'} />.</li>
        <li>Calculate the acceptance probability <InlineMath math={'\\alpha = \\min(1, \\frac{p(x^*)}{p(x_t)})'} />, where <InlineMath math={'p'} /> is the GMM density.</li>
        <li>Generate <InlineMath math={'u \\sim \\mathrm{Uniform}(0,1)'} />. If <InlineMath math={'u < \\alpha'} />, accept the proposal and set <InlineMath math={'x_{t+1} = x^*'} />. Otherwise, set <InlineMath math={'x_{t+1} = x_t'} />.</li>
        <li>Repeat steps 2-4 to generate a sequence of samples.</li>
      </ol>
    </div>
  );
} 