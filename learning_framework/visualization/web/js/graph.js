/**
 * Computational Graph Renderer using Canvas
 */
class GraphRenderer {
    constructor(canvasId = 'viz-canvas') {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.nodes = [];
        this.edges = [];
        this.selectedNode = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;

        if (this.canvas) {
            this.setupInteraction();
        }
    }

    render(graphData) {
        this.nodes = graphData.nodes || [];
        this.edges = graphData.edges || [];
        this.calculateLayout();
        this.draw();
    }

    calculateLayout() {
        // Auto-layout if positions not provided
        if (this.nodes.length > 0 && this.nodes[0].x === undefined) {
            this.autoLayout();
        }

        // Center the graph
        this.centerGraph();
    }

    autoLayout() {
        // Simple layered layout
        const layers = this.groupByLayers();
        const layerSpacing = 150;
        const nodeSpacing = 80;

        layers.forEach((layer, i) => {
            const x = i * layerSpacing + 100;
            layer.forEach((node, j) => {
                const y = (j - layer.length / 2) * nodeSpacing + this.canvas.height / 2;
                node.x = x;
                node.y = y;
            });
        });
    }

    groupByLayers() {
        // Group nodes by their topological order
        const layers = [];
        const visited = new Set();
        const nodeMap = new Map(this.nodes.map(n => [n.id, n]));

        // Find input nodes (no incoming edges)
        const incomingCount = new Map();
        this.nodes.forEach(n => incomingCount.set(n.id, 0));
        this.edges.forEach(e => {
            if (e.type === 'forward') {
                incomingCount.set(e.target, (incomingCount.get(e.target) || 0) + 1);
            }
        });

        let currentLayer = this.nodes.filter(n => incomingCount.get(n.id) === 0);

        while (currentLayer.length > 0) {
            layers.push(currentLayer);
            currentLayer.forEach(n => visited.add(n.id));

            // Find next layer
            const nextIds = new Set();
            this.edges.forEach(e => {
                if (e.type === 'forward' && visited.has(e.source) && !visited.has(e.target)) {
                    nextIds.add(e.target);
                }
            });

            currentLayer = [...nextIds].map(id => nodeMap.get(id)).filter(n => n);
        }

        return layers;
    }

    centerGraph() {
        if (this.nodes.length === 0) return;

        const minX = Math.min(...this.nodes.map(n => n.x));
        const maxX = Math.max(...this.nodes.map(n => n.x));
        const minY = Math.min(...this.nodes.map(n => n.y));
        const maxY = Math.max(...this.nodes.map(n => n.y));

        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;

        this.offsetX = (this.canvas.width - graphWidth) / 2 - minX;
        this.offsetY = (this.canvas.height - graphHeight) / 2 - minY;
    }

    draw() {
        if (!this.ctx) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Dark background
        ctx.fillStyle = '#16213e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);

        // Draw edges first
        this.edges.forEach(edge => this.drawEdge(edge));

        // Draw nodes on top
        this.nodes.forEach(node => this.drawNode(node));

        ctx.restore();
    }

    drawNode(node) {
        const ctx = this.ctx;
        const x = node.x;
        const y = node.y;
        const radius = 30;

        // Node colors based on type
        const colors = {
            input: '#4ecdc4',
            output: '#ff6b6b',
            hidden: '#00d9ff',
            matmul: '#ffd93d',
            relu: '#6bcb77',
            sigmoid: '#a66cff',
            softmax: '#ff9a3c',
            variable: '#00d9ff',
            default: '#16213e'
        };

        const color = colors[node.type] || colors.default;

        // Draw circle
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = this.selectedNode === node ? '#ffffff' : '#2a2a4a';
        ctx.lineWidth = this.selectedNode === node ? 3 : 2;
        ctx.stroke();

        // Draw label
        ctx.fillStyle = '#1a1a2e';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const lines = (node.label || node.type).split('\n');
        lines.forEach((line, i) => {
            const offset = (i - (lines.length - 1) / 2) * 12;
            ctx.fillText(line, x, y + offset);
        });
    }

    drawEdge(edge) {
        const ctx = this.ctx;
        const sourceNode = this.nodes.find(n => n.id === edge.source);
        const targetNode = this.nodes.find(n => n.id === edge.target);

        if (!sourceNode || !targetNode) return;

        const isGradient = edge.type === 'gradient';

        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);

        if (isGradient) {
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = '#ff6b6b';
        } else {
            ctx.setLineDash([]);
            ctx.strokeStyle = '#a0a0a0';
        }

        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw arrow
        this.drawArrow(sourceNode, targetNode, isGradient);
    }

    drawArrow(from, to, isGradient) {
        const ctx = this.ctx;
        const headLen = 10;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const angle = Math.atan2(dy, dx);

        // Shorten to not overlap with node
        const dist = Math.sqrt(dx * dx + dy * dy);
        const shortenBy = 35;
        const ratio = (dist - shortenBy) / dist;
        const endX = from.x + dx * ratio;
        const endY = from.y + dy * ratio;

        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
            endX - headLen * Math.cos(angle - Math.PI / 6),
            endY - headLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(endX, endY);
        ctx.lineTo(
            endX - headLen * Math.cos(angle + Math.PI / 6),
            endY - headLen * Math.sin(angle + Math.PI / 6)
        );

        ctx.strokeStyle = isGradient ? '#ff6b6b' : '#a0a0a0';
        ctx.stroke();
    }

    setupInteraction() {
        // Pan and zoom
        let isDragging = false;
        let lastX, lastY;

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                this.offsetX += e.clientX - lastX;
                this.offsetY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                this.draw();
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.scale *= delta;
            this.scale = Math.max(0.1, Math.min(3, this.scale));
            this.draw();
        });

        // Node selection
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - this.offsetX) / this.scale;
            const y = (e.clientY - rect.top - this.offsetY) / this.scale;

            this.selectedNode = this.findNodeAt(x, y);
            this.draw();

            if (this.selectedNode) {
                this.showNodeDetails(this.selectedNode);
            }
        });
    }

    findNodeAt(x, y) {
        const radius = 30;
        return this.nodes.find(node => {
            const dx = node.x - x;
            const dy = node.y - y;
            return Math.sqrt(dx * dx + dy * dy) < radius;
        });
    }

    showNodeDetails(node) {
        console.log('Selected node:', node);
        // Could show tooltip or panel with node details
    }
}

// Global instance
const graphRenderer = new GraphRenderer();
