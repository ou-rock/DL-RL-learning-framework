/**
 * Parameter controls UI and exploration
 */
class ParameterControls {
    constructor(containerId = 'param-panel') {
        this.container = document.getElementById(containerId);
        this.parameters = [];
        this.onChange = null;
    }

    render(config) {
        this.parameters = config.parameters || [];
        this.container.innerHTML = '';

        if (this.parameters.length === 0) {
            this.container.innerHTML = '<p>No parameters available</p>';
            return;
        }

        const title = document.createElement('h3');
        title.textContent = 'Parameters';
        this.container.appendChild(title);

        this.parameters.forEach(param => {
            const group = this.createParamGroup(param);
            this.container.appendChild(group);
        });
    }

    createParamGroup(param) {
        const group = document.createElement('div');
        group.className = 'param-group';

        // Label
        const labelRow = document.createElement('div');
        labelRow.className = 'param-label-row';

        const label = document.createElement('label');
        label.textContent = param.label || param.name;
        label.htmlFor = `param-${param.name}`;

        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'param-value';
        valueDisplay.id = `value-${param.name}`;
        valueDisplay.textContent = this.formatValue(param.value, param.scale);

        labelRow.appendChild(label);
        labelRow.appendChild(valueDisplay);

        // Slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `param-${param.name}`;
        slider.dataset.name = param.name;
        slider.dataset.scale = param.scale || 'linear';

        if (param.scale === 'log') {
            // Map to linear slider
            slider.min = Math.log10(param.min);
            slider.max = Math.log10(param.max);
            slider.step = (slider.max - slider.min) / 100;
            slider.value = Math.log10(param.value);
        } else {
            slider.min = param.min;
            slider.max = param.max;
            slider.step = param.step || (param.max - param.min) / 100;
            slider.value = param.value;
        }

        slider.addEventListener('input', (e) => this.handleSliderChange(e));

        group.appendChild(labelRow);
        group.appendChild(slider);

        return group;
    }

    handleSliderChange(event) {
        const slider = event.target;
        const name = slider.dataset.name;
        const scale = slider.dataset.scale;

        let value = parseFloat(slider.value);
        if (scale === 'log') {
            value = Math.pow(10, value);
        }

        // Update display
        const valueEl = document.getElementById(`value-${name}`);
        if (valueEl) {
            valueEl.textContent = this.formatValue(value, scale);
        }

        // Notify change
        if (this.onChange) {
            this.onChange(name, value);
        }
    }

    formatValue(value, scale) {
        if (scale === 'log' || value < 0.01) {
            return value.toExponential(2);
        }
        return value.toFixed(3);
    }

    setOnChange(callback) {
        this.onChange = callback;
    }

    getValues() {
        const values = {};
        this.parameters.forEach(param => {
            const slider = document.getElementById(`param-${param.name}`);
            if (slider) {
                let value = parseFloat(slider.value);
                if (param.scale === 'log') {
                    value = Math.pow(10, value);
                }
                values[param.name] = value;
            }
        });
        return values;
    }

    setValue(name, value) {
        const slider = document.getElementById(`param-${name}`);
        const param = this.parameters.find(p => p.name === name);

        if (slider && param) {
            if (param.scale === 'log') {
                slider.value = Math.log10(value);
            } else {
                slider.value = value;
            }

            const valueEl = document.getElementById(`value-${name}`);
            if (valueEl) {
                valueEl.textContent = this.formatValue(value, param.scale);
            }
        }
    }
}

/**
 * Loss landscape visualization
 */
class LossLandscape {
    constructor(canvasId = 'viz-canvas') {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    }

    render(effectData) {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const { values, losses, param_name } = effectData;

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Background
        ctx.fillStyle = '#16213e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Find bounds
        const validLosses = losses.filter(l => !isNaN(l) && isFinite(l));
        if (validLosses.length === 0) return;

        const minLoss = Math.min(...validLosses);
        const maxLoss = Math.max(...validLosses);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);

        // Padding
        const padding = { top: 40, right: 40, bottom: 60, left: 80 };
        const width = this.canvas.width - padding.left - padding.right;
        const height = this.canvas.height - padding.top - padding.bottom;

        // Draw axes
        ctx.strokeStyle = '#a0a0a0';
        ctx.lineWidth = 1;

        // Y axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + height);
        ctx.stroke();

        // X axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top + height);
        ctx.lineTo(padding.left + width, padding.top + height);
        ctx.stroke();

        // Draw curve
        ctx.beginPath();
        ctx.strokeStyle = '#00d9ff';
        ctx.lineWidth = 2;

        let started = false;
        values.forEach((val, i) => {
            const loss = losses[i];
            if (isNaN(loss) || !isFinite(loss)) return;

            const x = padding.left + ((val - minVal) / (maxVal - minVal || 1)) * width;
            const y = padding.top + height - ((loss - minLoss) / (maxLoss - minLoss || 1)) * height;

            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Labels
        ctx.fillStyle = '#e8e8e8';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(param_name || 'Parameter', padding.left + width / 2, this.canvas.height - 10);

        ctx.save();
        ctx.translate(20, padding.top + height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Loss', 0, 0);
        ctx.restore();

        // Value labels
        ctx.font = '10px sans-serif';
        ctx.fillText(minVal.toExponential(1), padding.left, this.canvas.height - 30);
        ctx.fillText(maxVal.toExponential(1), padding.left + width, this.canvas.height - 30);
    }
}

// Global instances
const paramControls = new ParameterControls();
const lossLandscape = new LossLandscape();
