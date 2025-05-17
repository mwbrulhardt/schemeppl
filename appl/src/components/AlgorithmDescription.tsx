import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

export default function AlgorithmDescription() {
  return (
    <div className="bg-white p-6 rounded-lg shadow mb-8">
      <h2 className="text-xl font-semibold mb-4">Algorithm Description</h2>
      <p className="mb-4">
        This implementation uses Metropolis-Hastings MCMC to infer the two
        component means of a Gaussian Mixture Model (GMM) with known variances
        and mixture weight from observed data. The model assumes the data is
        generated from a mixture of two normal distributions:
      </p>
      <div className="mb-4 text-center">
        <BlockMath
          math={
            'p(x|\\boldsymbol{\\mu},\\boldsymbol{\\sigma},\\pi) = \\pi \\cdot N(x|\\mu_1,\\sigma_1^2) + (1-\\pi) \\cdot N(x|\\mu_2,\\sigma_2^2)'
          }
        />
      </div>
      <p className="mb-4">where:</p>
      <ul className="list-disc pl-6 mb-4">
        <li>
          <InlineMath math={'\\pi \\in (0,1)'} /> is the mixture weight
        </li>
        <li>
          <InlineMath math={'\\boldsymbol{\\mu} \\in \\mathbb{R}^2'} /> are the
          means of the two components, where{' '}
          <InlineMath math={'\\boldsymbol{\\mu} = (\\mu_1,\\mu_2)'} />
        </li>
        <li>
          <InlineMath math={'\\sigma_1^2, \\sigma_2^2 \\in \\mathbb{R}_{>0}'} />{' '}
          are the variances of the two components
        </li>
      </ul>
      <p className="mb-4">
        Assume <InlineMath math={'\\sigma_1^2, \\sigma_2^2, \\pi'} /> are known.
        The algorithm works as follows:
      </p>
      <ol className="list-decimal pl-6">
        <li>
          Start with initial values for{' '}
          <InlineMath math={'\\boldsymbol{\\mu}_0 \\in \\mathbb{R}^2'} />, with
          the constraint that <InlineMath math={'\\mu^0_1 < \\mu^0_2'} />.
        </li>
        <li>
          At each step <InlineMath math={'n \\in \\mathbb{N}'} />, propose new
          values by sampling:
        </li>
        <div className="ml-8 mb-4">
          <BlockMath
            math={
              '\\boldsymbol{\\mu}^* \\sim \\mathcal{N}\\bigl(\\boldsymbol{\\mu}_n, \\operatorname{diag}(\\tau_1^2,\\tau_2^2)\\bigr)'
            }
          />
        </div>
        <p className="ml-8 mb-4">
          where{' '}
          <InlineMath math={'\\tau_1^2, \\tau_2^2 \\in \\mathbb{R}_{>0}'} /> are
          the proposal variances.
        </p>
        <li>
          Calculate the acceptance probability based on the likelihood of the
          observed data <InlineMath math={'\\mathbf{x} \\in \\mathbb{R}^N'} />:
        </li>
        , where <InlineMath math={'N'} /> is the number of observations.
        <div className="ml-8 mb-4">
          <BlockMath
            math={
              '\\alpha = \\min\\biggl(1, \\frac{p(\\boldsymbol{\\mu}^* | \\mathbf{x})}{p(\\boldsymbol{\\mu} | \\mathbf{x})}\\biggr)'
            }
          />
        </div>
        <li>
          Generate <InlineMath math={'u \\sim \\mathrm{Uniform}(0,1)'} />. If{' '}
          <InlineMath math={'u < \\alpha'} />, accept the proposal. Otherwise,
          keep the current values.
        </li>
        <li>
          Repeat steps 2-4 to generate a sequence of samples from the posterior
          distribution of the parameters.
        </li>
      </ol>
      <p className="mt-4">
        The simulation tracks the acceptance ratio and provides visualization of
        the sampling process, including:
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li>The trace of accepted and rejected proposals for both means</li>
        <li>The posterior distribution of the parameters</li>
        <li>The fit of the GMM to the observed data</li>
      </ul>

      <h3 className="text-lg font-semibold mt-6 mb-3">
        Model Specification with{' '}
        <InlineMath math={'\\sigma_1^2 = \\sigma_2^2 = 1'} /> and{' '}
        <InlineMath math={'\\pi = 0.5'} />
      </h3>
      <pre className="bg-gray-50 p-4 rounded overflow-auto text-sm">
        {`;; prior on component means
(sample mu1 (normal 0.0 1.0))
(sample mu2 (normal 0.0 1.0))

;; identifiability constraint: add 0 to log-p if ordered, −∞ otherwise
(constrain (< mu1 mu2))

(define p 0.5)
(define mix (mixture (list (normal mu1 1.0) (normal mu2 1.0)) (list p (- 1.0 p))))

(define observe-point (lambda (x) (observe (gensym) mix x)))

(for-each observe-point data)`}
      </pre>
    </div>
  );
}
