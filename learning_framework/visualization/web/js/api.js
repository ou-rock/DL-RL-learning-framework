/**
 * API client for communicating with visualization server
 * Supports both live backend and static JSON fallback for Vercel deployment
 */
class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
        this.useStatic = false; // Will be set to true if backend is unavailable
        this.staticData = {};   // Cache for static JSON data
    }

    async fetchJSON(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return response.json();
    }

    async fetchStatic(path) {
        if (this.staticData[path]) {
            return this.staticData[path];
        }
        const response = await fetch(`${this.baseUrl}/data/${path}`);
        if (!response.ok) {
            throw new Error(`Static file not found: ${path}`);
        }
        const data = await response.json();
        this.staticData[path] = data;
        return data;
    }

    async getConcepts() {
        try {
            if (!this.useStatic) {
                const data = await this.fetchJSON('/api/concepts');
                return data.concepts || [];
            }
        } catch (e) {
            console.log('Backend unavailable, using static data');
            this.useStatic = true;
        }
        // Fallback to static JSON
        const data = await this.fetchStatic('concepts.json');
        return data.concepts || [];
    }

    async getConceptDetail(slug) {
        if (this.useStatic) {
            // Return mock detail from concepts list
            const data = await this.fetchStatic('concepts.json');
            const concept = data.concepts.find(c => c.slug === slug);
            return concept ? {
                ...concept,
                visualizations: ['computational_graph'],
                has_quiz: true,
                has_challenge: true
            } : null;
        }
        return this.fetchJSON(`/api/concept/${slug}`);
    }

    async getGraphData(concept = null) {
        if (this.useStatic) {
            return this.fetchStatic('graph.json');
        }
        const endpoint = concept ? `/api/graph?concept=${concept}` : '/api/graph';
        return this.fetchJSON(endpoint);
    }

    async getVisualizationData(concept, vizName = 'main_visualization') {
        if (this.useStatic) {
            const graphData = await this.fetchStatic('graph.json');
            return {
                type: 'graph',
                ...graphData
            };
        }
        return this.fetchJSON(`/api/viz/${concept}/${vizName}`);
    }
}

// WebSocket manager for real-time updates
class WebSocketManager {
    constructor(url = null) {
        this.url = url || `ws://${window.location.host}/ws`;
        this.ws = null;
        this.handlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this._updateStatus(true);
                this._emit('connected');
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this._updateStatus(false);
                this._emit('disconnected');
                this._tryReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this._emit('error', error);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._emit(data.type, data.payload);
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            this._tryReconnect();
        }
    }

    send(type, payload) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, payload }));
        }
    }

    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
    }

    _emit(event, data) {
        const handlers = this.handlers[event] || [];
        handlers.forEach(handler => handler(data));
    }

    _updateStatus(connected) {
        const statusEl = document.getElementById('ws-status');
        if (statusEl) {
            statusEl.textContent = connected ? 'Connected' : 'Disconnected';
            statusEl.classList.toggle('connected', connected);
        }
    }

    _tryReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        }
    }
}

// Global instances
const api = new API();
const wsManager = new WebSocketManager();
