/**
 * Real-time training visualization
 */
class TrainingRenderer {
    constructor(canvasId = 'viz-canvas') {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.data = null;
        this.animationFrame = null;
    }

    render(data) {
        this.data = data;
        this.draw();
    }

    update(updateData) {
        if (!this.data) {
            this.data = {
                history: { epochs: [], loss: [], accuracy: [] },
                smoothed_loss: [],
                current: {}
            };
        }

        // Append new data point
        const { epoch, loss, accuracy } = updateData;
        this.data.history.epochs.push(epoch);
        this.data.history.loss.push(loss);
        this.data.history.accuracy.push(accuracy);
        this.data.current = updateData;

        this.draw();
    }

    draw() {
        if (!this.ctx || !this.data) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Background
        ctx.fillStyle = '#16213e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const { history, smoothed_loss, current, total_epochs } = this.data;

        // Layout
        const padding = { top: 60, right: 40, bottom: 60, left: 80 };
        const width = this.canvas.width - padding.left - padding.right;
        const height = (this.canvas.height - padding.top - padding.bottom - 40) / 2;

        // Draw title
        ctx.fillStyle = '#e8e8e8';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Training Progress', this.canvas.width / 2, 25);

        // Draw stats
        if (current && current.epoch !== undefined) {
            ctx.font = '12px sans-serif';
            const lossStr = current.loss?.toFixed(4) || '-';
            const accStr = current.accuracy !== undefined ? (current.accuracy * 100).toFixed(1) : '-';
            const statsText = `Epoch: ${current.epoch}/${total_epochs || '?'} | Loss: ${lossStr} | Acc: ${accStr}%`;
            ctx.fillText(statsText, this.canvas.width / 2, 45);
        }

        // Draw loss chart
        if (history && history.epochs) {
            this.drawChart(
                padding.left,
                padding.top,
                width,
                height,
                history.epochs,
                history.loss,
                smoothed_loss,
                'Loss',
                '#ff6b6b',
                '#ff6b6b80'
            );

            // Draw accuracy chart
            this.drawChart(
                padding.left,
                padding.top + height + 40,
                width,
                height,
                history.epochs,
                history.accuracy,
                null,
                'Accuracy',
                '#4ecdc4',
                null
            );
        }
    }

    drawChart(x, y, width, height, epochs, values, smoothed, label, color, smoothColor) {
        const ctx = this.ctx;

        if (!epochs || !epochs.length) return;

        // Find bounds
        const validValues = values.filter(v => !isNaN(v) && isFinite(v));
        if (!validValues.length) return;

        let minVal = Math.min(...validValues);
        let maxVal = Math.max(...validValues);

        // Add padding to range
        const range = maxVal - minVal || 1;
        minVal -= range * 0.05;
        maxVal += range * 0.05;

        const minEpoch = Math.min(...epochs);
        const maxEpoch = Math.max(...epochs);
        const epochRange = maxEpoch - minEpoch || 1;

        // Draw axes
        ctx.strokeStyle = '#4a4a6a';
        ctx.lineWidth = 1;

        // Y axis
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + height);
        ctx.stroke();

        // X axis
        ctx.beginPath();
        ctx.moveTo(x, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.stroke();

        // Draw smoothed line first (if provided)
        if (smoothed && smoothed.length) {
            ctx.beginPath();
            ctx.strokeStyle = smoothColor;
            ctx.lineWidth = 3;

            let started = false;
            smoothed.forEach((val, i) => {
                if (isNaN(val) || !isFinite(val)) return;

                const px = x + ((epochs[i] - minEpoch) / epochRange) * width;
                const py = y + height - ((val - minVal) / (maxVal - minVal)) * height;

                if (!started) {
                    ctx.moveTo(px, py);
                    started = true;
                } else {
                    ctx.lineTo(px, py);
                }
            });

            ctx.stroke();
        }

        // Draw main line
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        let started = false;
        values.forEach((val, i) => {
            if (isNaN(val) || !isFinite(val)) return;

            const px = x + ((epochs[i] - minEpoch) / epochRange) * width;
            const py = y + height - ((val - minVal) / (maxVal - minVal)) * height;

            if (!started) {
                ctx.moveTo(px, py);
                started = true;
            } else {
                ctx.lineTo(px, py);
            }
        });

        ctx.stroke();

        // Draw current point
        if (values.length) {
            const lastVal = values[values.length - 1];
            const lastEpoch = epochs[epochs.length - 1];
            if (!isNaN(lastVal) && isFinite(lastVal)) {
                const px = x + ((lastEpoch - minEpoch) / epochRange) * width;
                const py = y + height - ((lastVal - minVal) / (maxVal - minVal)) * height;

                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        }

        // Labels
        ctx.fillStyle = '#a0a0a0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', x + width / 2, y + height + 25);

        ctx.save();
        ctx.translate(x - 40, y + height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(label, 0, 0);
        ctx.restore();

        // Value labels
        ctx.textAlign = 'right';
        ctx.fillText(maxVal.toFixed(3), x - 5, y + 10);
        ctx.fillText(minVal.toFixed(3), x - 5, y + height);
    }
}

// Global instance
const trainingRenderer = new TrainingRenderer();
