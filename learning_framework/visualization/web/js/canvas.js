/**
 * Canvas renderer for general visualizations
 * Placeholder - will be expanded in Task 4 (Parameter Exploration)
 */
class CanvasRenderer {
    constructor(canvasId = 'viz-canvas') {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    }

    render(data) {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.fillStyle = '#16213e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Show placeholder message
        this.ctx.fillStyle = '#e8e8e8';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Visualization loading...', this.canvas.width / 2, this.canvas.height / 2);
    }

    clear() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Global instance
const canvasRenderer = new CanvasRenderer();
