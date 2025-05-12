import { useEffect, useRef } from 'react';
import { SimulationState, Parameters } from '@/hooks/useSimulator';

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

  // Draw the walk visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state?.steps?.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Set up coordinate system
      const padding = 40;
      const width = canvas.width - 2 * padding;
      const height = canvas.height - 2 * padding;

      // Calculate domain based on means
      const xMin = Math.min(parameters.mean1, parameters.mean2) - 2;
      const xMax = Math.max(parameters.mean1, parameters.mean2) + 2;
      const yMin = xMin; // Use same range for y-axis
      const yMax = xMax;

      // Draw axes
      ctx.beginPath();
      ctx.strokeStyle = '#ccc';
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, canvas.height - padding);
      ctx.lineTo(canvas.width - padding, canvas.height - padding);
      ctx.stroke();

      // Draw grid
      ctx.strokeStyle = '#eee';
      const numGridLines = 10;
      for (let i = 0; i <= numGridLines; i++) {
        const x = padding + (width * i) / numGridLines;
        const y = padding + (height * i) / numGridLines;
        
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, canvas.height - padding);
        ctx.stroke();

        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();

        // Draw axis labels
        const xValue = xMin + (xMax - xMin) * (i / numGridLines);
        const yValue = yMin + (yMax - yMin) * (i / numGridLines);
        
        // X-axis labels
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(xValue.toFixed(1), x, canvas.height - padding + 20);
        
        // Y-axis labels
        ctx.textAlign = 'right';
        ctx.fillText(yValue.toFixed(1), padding - 5, canvas.height - y);
      }

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

      // Helper function to convert from data coordinates to canvas coordinates
      const toCanvasX = (x: number) => padding + width * (x - xMin) / (xMax - xMin);
      const toCanvasY = (y: number) => canvas.height - padding - height * (y - yMin) / (yMax - yMin);

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
    <div className="w-full">
      <canvas
        ref={canvasRef}
        width={1200}
        height={600}
        className="w-full h-auto border border-gray-200 rounded-lg"
      />
    </div>
  );
} 