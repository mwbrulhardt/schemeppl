import { Parameters, SimulationState } from '@/hooks/useSimulator';
import { useEffect, useRef } from 'react';

interface Point {
  x: number;
  y: number;
  weight: number;
  accepted: boolean;
}

interface WalkVisualizationProps {
  state: SimulationState;
  isRunning: boolean;
  parameters: Parameters;
}

export default function WalkVisualization({ state, isRunning, parameters }: WalkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const pointsRef = useRef<Point[]>([]);
  const lastStepRef = useRef<{ mu1: number; mu2: number; accepted: boolean } | null>(null);

  // Initialize canvas with proper DPI scaling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    // Set the canvas size accounting for DPI
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale the context to match the DPI
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
  }, []);

  // Force clear the visualization when state is null or steps are empty (after reset)
  useEffect(() => {
    if (!state) {
      console.log("Force clearing visualization points only - state is null (full reset)");
      pointsRef.current = [];
      lastStepRef.current = null;
      
      // Redraw canvas with grid but no points
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          drawGridAndAxes(ctx, canvas, parameters);
        }
      }
      return;
    }
    
    if (state.steps.length === 0 && pointsRef.current.length > 0) {
      console.log("Force clearing visualization points only - steps array is empty");
      pointsRef.current = [];
      lastStepRef.current = null;
      
      // Redraw canvas with grid but no points
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          drawGridAndAxes(ctx, canvas, parameters);
        }
      }
    }
  }, [state, parameters]);

  // Helper function to draw grid and axes only (no points)
  const drawGridAndAxes = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, parameters: Parameters) => {
    const padding = 40;
    const width = canvas.width / (window.devicePixelRatio || 1) - 2 * padding;
    const height = canvas.height / (window.devicePixelRatio || 1) - 2 * padding;
    const xMin = Math.min(parameters.mu1, parameters.mu2) - 2;
    const xMax = Math.max(parameters.mu1, parameters.mu2) + 2;
    const yMin = xMin;
    const yMax = xMax;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set up text properties
    ctx.font = '12px Arial';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';

    // Draw x-axis
    ctx.beginPath();
    ctx.moveTo(padding, canvas.height / (window.devicePixelRatio || 1) - padding);
    ctx.lineTo(canvas.width / (window.devicePixelRatio || 1) - padding, canvas.height / (window.devicePixelRatio || 1) - padding);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();

    // Draw y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height / (window.devicePixelRatio || 1) - padding);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();

    // Draw grid lines and labels
    const numTicks = 5;
    const xStep = (xMax - xMin) / numTicks;
    const yStep = (yMax - yMin) / numTicks;

    for (let i = 0; i <= numTicks; i++) {
      const x = xMin + i * xStep;
      const y = yMin + i * yStep;
      
      // X-axis ticks and labels
      const xPos = padding + (x - xMin) / (xMax - xMin) * width;
      ctx.beginPath();
      ctx.moveTo(xPos, canvas.height / (window.devicePixelRatio || 1) - padding);
      ctx.lineTo(xPos, canvas.height / (window.devicePixelRatio || 1) - padding + 5);
      ctx.stroke();
      ctx.fillText(x.toFixed(1), xPos, canvas.height / (window.devicePixelRatio || 1) - padding + 20);

      // Y-axis ticks and labels
      const yPos = canvas.height / (window.devicePixelRatio || 1) - padding - (y - yMin) / (yMax - yMin) * height;
      ctx.beginPath();
      ctx.moveTo(padding, yPos);
      ctx.lineTo(padding - 5, yPos);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(y.toFixed(1), padding - 10, yPos + 4);
      ctx.textAlign = 'center';
    }

    // Add axis labels
    ctx.font = '14px Arial';
    ctx.fillStyle = '#333';
    
    // X-axis label (μ₁)
    ctx.fillText('μ₁', canvas.width / (2 * (window.devicePixelRatio || 1)), canvas.height / (window.devicePixelRatio || 1) - 10);
    
    // Y-axis label (μ₂)
    ctx.save();
    ctx.translate(20, canvas.height / (2 * (window.devicePixelRatio || 1)));
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('μ₂', 0, 0);
    ctx.restore();
  };

  // Draw the walk visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Handle null state
    if (!state) {
      console.log("Main drawing effect: state is null, redrawing grid only");
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawGridAndAxes(ctx, canvas, parameters);
      }
      return;
    }
    
    // Check if steps array is empty but we have points - this means a reset
    if (state.steps.length === 0 && pointsRef.current.length > 0) {
      console.log("Clearing walk visualization points after reset");
      pointsRef.current = [];
      lastStepRef.current = null;
      
      // Draw grid but no points
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawGridAndAxes(ctx, canvas, parameters);
      }
      return;
    }
    
    if (!state.steps.length) {
      // Just draw grid if no steps
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawGridAndAxes(ctx, canvas, parameters);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw grid and axes first
      drawGridAndAxes(ctx, canvas, parameters);

      // Helper function to convert from data coordinates to canvas coordinates
      const padding = 40;
      const width = canvas.width / (window.devicePixelRatio || 1) - 2 * padding;
      const height = canvas.height / (window.devicePixelRatio || 1) - 2 * padding;
      const xMin = Math.min(parameters.mu1, parameters.mu2) - 2;
      const xMax = Math.max(parameters.mu1, parameters.mu2) + 2;
      const yMin = xMin;
      const yMax = xMax;
      
      const toCanvasX = (x: number) => padding + width * (x - xMin) / (xMax - xMin);
      const toCanvasY = (y: number) => canvas.height / (window.devicePixelRatio || 1) - padding - height * (y - yMin) / (yMax - yMin);

      // Update points array if we have a new step
      const lastStep = state.steps[state.steps.length - 1];
      if (lastStepRef.current !== lastStep) {
        // Fade out only rejected points
        pointsRef.current = pointsRef.current.map(p => ({
          ...p,
          weight: p.accepted ? 1 : p.weight * 0.95 // Keep accepted points at full opacity
        }));

        // Add new point
        pointsRef.current.push({
          x: lastStep.mu1,
          y: lastStep.mu2,
          weight: 1,
          accepted: lastStep.accepted
        });

        // Remove only rejected points that are too faint
        pointsRef.current = pointsRef.current.filter(p => p.accepted || p.weight > 0.01);
        
        lastStepRef.current = lastStep;
      }

      // Draw points
      pointsRef.current.forEach(point => {
        const x = toCanvasX(point.x);
        const y = toCanvasY(point.y);
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = point.accepted ? 
          `rgba(128, 128, 128, ${point.weight})` : // Gray for accepted points
          `rgba(255, 0, 0, ${point.weight})`; // Red for rejected points
        ctx.fill();
      });

      // Draw current point and proposed move
      const current = state.steps[state.steps.length - 1];
      const x1 = toCanvasX(current.mu1);
      const y1 = toCanvasY(current.mu2);

      // Draw current point
      ctx.beginPath();
      ctx.arc(x1, y1, 5, 0, Math.PI * 2);
      ctx.fillStyle = current.accepted ? '#808080' : '#ff0000'; // Gray for accepted, red for rejected
      ctx.fill();

      // Draw arrow from previous point to proposed point
      if (state.steps.length > 1) {
        const prev = state.steps[state.steps.length - 2];
        const curr = state.steps[state.steps.length - 1];
        // If accepted, the chain moved to curr; if rejected, the chain stayed at prev, but proposal was to curr
        // The arrow should always point from prev to curr (the proposal)
        const xPrev = toCanvasX(prev.mu1);
        const yPrev = toCanvasY(prev.mu2);
        const xCurr = toCanvasX(curr.mu1);
        const yCurr = toCanvasY(curr.mu2);

        // Only draw the arrow if the proposal is to a different point
        if (prev.mu1 !== curr.mu1 || prev.mu2 !== curr.mu2) {
          // Draw arrow line
          ctx.beginPath();
          ctx.moveTo(xPrev, yPrev);
          ctx.lineTo(xCurr, yCurr);
          ctx.strokeStyle = curr.accepted ? '#808080' : '#ff0000';
          ctx.stroke();

          // Draw arrowhead at the proposed point (curr)
          const angle = Math.atan2(yCurr - yPrev, xCurr - xPrev);
          const arrowLength = 10;
          const arrowAngle = Math.PI / 6;

          ctx.beginPath();
          ctx.moveTo(xCurr, yCurr);
          ctx.lineTo(
            xCurr - arrowLength * Math.cos(angle - arrowAngle),
            yCurr - arrowLength * Math.sin(angle - arrowAngle)
          );
          ctx.lineTo(
            xCurr - arrowLength * Math.cos(angle + arrowAngle),
            yCurr - arrowLength * Math.sin(angle + arrowAngle)
          );
          ctx.closePath();
          ctx.fillStyle = curr.accepted ? '#808080' : '#ff0000';
          ctx.fill();
        }
      }

      if (isRunning) {
        animationFrameRef.current = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [state?.steps, isRunning, parameters]);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4">Parameter Space Walk</h2>
      <div className="flex justify-center">
        <canvas
          ref={canvasRef}
          style={{ width: '800px', height: '600px' }}
          className="max-w-full h-auto border border-gray-200 rounded-lg"
        />
      </div>
    </div>
  );
} 