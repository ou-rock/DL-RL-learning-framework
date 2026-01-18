/**
 * API client for communicating with visualization server
 */
class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
    }

    async fetchJSON(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return response.json();
    }

    async getConcepts() {
        const data = await this.fetchJSON('/api/concepts');
        return data.concepts || [];
    }

    async getConceptDetail(slug) {
        return this.fetchJSON(`/api/concept/${slug}`);
    }

    async getGraphData(concept = null) {
        const endpoint = concept ? `/api/graph?concept=${concept}` : '/api/graph';
        return this.fetchJSON(endpoint);
    }

    async getVisualizationData(concept, vizName = 'main_visualization') {
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
