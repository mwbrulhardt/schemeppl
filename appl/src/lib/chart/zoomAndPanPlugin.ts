import type { Chart } from 'chart.js';

// Extend Chart type to include our custom property
interface ChartWithWheel extends Chart {
  _removeListeners?: () => void;
}

/**
 * Zoom & pan plugin for linear X & Y scales.
 * – wheel-down  ➜ zoom-in  (10 %)
 * – wheel-up    ➜ zoom-out (10 %)
 * – mouse drag  ➜ pan chart
 * Hold ⌘ / Ctrl to bypass wheel zoom (page zoom).
 */
export const zoomAndPanPlugin = {
  id: 'wheel-zoom',
  beforeInit(chart: Chart<'scatter'>) {
    // Variables for dragging functionality
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    function onWheel(e: WheelEvent) {
      if (e.ctrlKey || e.metaKey) return; // let browser pinch-zoom
      e.preventDefault();

      const scales = [chart.scales.x, chart.scales.y];
      const factor = e.deltaY > 0 ? 0.9 : 1.1; // <1 = zoom-in

      scales.forEach((scale) => {
        const mid = (scale.min! + scale.max!) / 2;
        const range = (scale.max! - scale.min!) * factor;
        scale.options.min = mid - range / 2;
        scale.options.max = mid + range / 2;
      });

      chart.update('none'); // instant redraw
    }

    function onMouseDown(e: MouseEvent) {
      if (e.button !== 0) return; // Only respond to left mouse button

      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;

      // Change cursor to indicate dragging
      chart.canvas.style.cursor = 'grabbing';
    }

    function onMouseMove(e: MouseEvent) {
      if (!isDragging) return;

      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;

      const xScale = chart.scales.x;
      const yScale = chart.scales.y;

      // Calculate pixel-to-data ratio
      const xRange = xScale.max! - xScale.min!;
      const yRange = yScale.max! - yScale.min!;
      const xPixelRatio = xRange / xScale.width;
      const yPixelRatio = yRange / yScale.height;

      // Move in the opposite direction of mouse movement
      if (
        typeof xScale.options.min === 'number' &&
        typeof xScale.options.max === 'number'
      ) {
        xScale.options.min -= dx * xPixelRatio;
        xScale.options.max -= dx * xPixelRatio;
      }

      if (
        typeof yScale.options.min === 'number' &&
        typeof yScale.options.max === 'number'
      ) {
        yScale.options.min += dy * yPixelRatio; // Y axis is inverted in canvas
        yScale.options.max += dy * yPixelRatio;
      }

      chart.update('none');
    }

    function onMouseUp() {
      isDragging = false;
      chart.canvas.style.cursor = 'default';
    }

    function onMouseLeave() {
      if (isDragging) {
        isDragging = false;
        chart.canvas.style.cursor = 'default';
      }
    }

    // Add event listeners
    chart.canvas.addEventListener('wheel', onWheel, { passive: false });
    chart.canvas.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    chart.canvas.addEventListener('mouseleave', onMouseLeave);

    // tidy-up on chart.destroy()
    (chart as ChartWithWheel)._removeListeners = () => {
      chart.canvas.removeEventListener('wheel', onWheel);
      chart.canvas.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      chart.canvas.removeEventListener('mouseleave', onMouseLeave);
    };
  },
  beforeDestroy(chart: Chart) {
    (chart as ChartWithWheel)._removeListeners?.();
  },
};
