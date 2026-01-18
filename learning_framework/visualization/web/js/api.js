/**
 * API client for communicating with visualization server
 * Supports both live backend and static JSON fallback for Vercel deployment
 *
 * Features:
 * - Anonymous user tracking via localStorage UUID
 * - Progress saving with offline queue
 * - Automatic retry for failed requests
 */

// User ID management
const UserManager = {
    STORAGE_KEY: 'learning_framework_user_id',

    getUserId() {
        let userId = localStorage.getItem(this.STORAGE_KEY);
        if (!userId) {
            userId = crypto.randomUUID();
            localStorage.setItem(this.STORAGE_KEY, userId);
            console.log('Generated new user ID:', userId);
        }
        return userId;
    },

    clearUserId() {
        localStorage.removeItem(this.STORAGE_KEY);
    }
};

// Offline progress queue for reliability
class ProgressQueue {
    constructor() {
        this.STORAGE_KEY = 'learning_framework_progress_queue';
    }

    add(progressData) {
        const queue = this.getQueue();
        queue.push({
            ...progressData,
            timestamp: Date.now()
        });
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(queue));
    }

    getQueue() {
        try {
            return JSON.parse(localStorage.getItem(this.STORAGE_KEY)) || [];
        } catch {
            return [];
        }
    }

    remove(index) {
        const queue = this.getQueue();
        queue.splice(index, 1);
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(queue));
    }

    clear() {
        localStorage.removeItem(this.STORAGE_KEY);
    }

    async flush(api) {
        const queue = this.getQueue();
        if (queue.length === 0) return;

        console.log(`Flushing ${queue.length} queued progress updates...`);

        for (let i = queue.length - 1; i >= 0; i--) {
            const item = queue[i];
            try {
                await api.saveProgress(item.concept, item.score, true);
                this.remove(i);
            } catch (e) {
                console.warn('Failed to flush progress item:', e);
                // Stop flushing if we hit an error
                break;
            }
        }
    }
}

class API {
    constructor(baseUrl = '') {
        // Allow override via data attribute or environment
        const configuredUrl = document.querySelector('meta[name="api-base-url"]')?.content;
        this.baseUrl = baseUrl || configuredUrl || window.location.origin;
        this.useStatic = false; // Will be set to true if backend is unavailable
        this.staticData = {};   // Cache for static JSON data
        this.progressQueue = new ProgressQueue();

        // Try to flush any queued progress on initialization
        this._tryFlushQueue();
    }

    _getHeaders() {
        return {
            'Content-Type': 'application/json',
            'X-User-ID': UserManager.getUserId()
        };
    }

    async _tryFlushQueue() {
        // Delay slightly to let the page load
        setTimeout(() => {
            this.progressQueue.flush(this).catch(e => {
                console.log('Queue flush deferred:', e.message);
            });
        }, 2000);
    }

    async fetchJSON(endpoint, options = {}) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers: {
                ...this._getHeaders(),
                ...options.headers
            }
        });
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
            // Try to load concept-specific data first
            try {
                const conceptData = await this.fetchStatic(`concepts/${slug}.json`);
                return {
                    ...conceptData,
                    visualizations: ['computational_graph'],
                    has_quiz: true,
                    has_challenge: true
                };
            } catch (e) {
                // Fallback to concepts list
                const data = await this.fetchStatic('concepts.json');
                const concept = data.concepts.find(c => c.slug === slug);
                return concept ? {
                    ...concept,
                    visualizations: ['computational_graph'],
                    has_quiz: true,
                    has_challenge: true
                } : null;
            }
        }
        return this.fetchJSON(`/api/concept/${slug}`);
    }

    async getGraphData(concept = null) {
        if (this.useStatic) {
            // Try concept-specific graph first
            if (concept) {
                try {
                    const conceptData = await this.fetchStatic(`concepts/${concept}.json`);
                    if (conceptData.graph) {
                        return conceptData.graph;
                    }
                } catch (e) {
                    // Fall through to default
                }
            }
            return this.fetchStatic('graph.json');
        }
        const endpoint = concept ? `/api/graph?concept=${concept}` : '/api/graph';
        return this.fetchJSON(endpoint);
    }

    async getVisualizationData(concept, vizName = 'main_visualization') {
        if (this.useStatic) {
            // Try concept-specific visualization first
            try {
                const conceptData = await this.fetchStatic(`concepts/${concept}.json`);
                if (conceptData.graph) {
                    return {
                        type: 'graph',
                        ...conceptData.graph
                    };
                }
            } catch (e) {
                // Fall through to default
            }
            const graphData = await this.fetchStatic('graph.json');
            return {
                type: 'graph',
                ...graphData
            };
        }
        return this.fetchJSON(`/api/viz/${concept}/${vizName}`);
    }

    /**
     * Save user progress after completing a quiz
     * @param {string} concept - Concept slug
     * @param {number} score - Score from 0.0 to 1.0
     * @param {boolean} skipQueue - If true, don't add to queue on failure
     * @returns {Promise<{success: boolean}>}
     */
    async saveProgress(concept, score, skipQueue = false) {
        if (this.useStatic) {
            // Can't save in static mode - queue for later
            if (!skipQueue) {
                this.progressQueue.add({ concept, score });
                console.log('Progress queued (static mode):', concept);
            }
            return { success: false, queued: true };
        }

        try {
            const result = await this.fetchJSON('/api/progress', {
                method: 'POST',
                body: JSON.stringify({ concept, score })
            });
            return result;
        } catch (e) {
            console.error('Failed to save progress:', e);
            if (!skipQueue) {
                this.progressQueue.add({ concept, score });
                console.log('Progress queued for retry:', concept);
            }
            return { success: false, queued: true, error: e.message };
        }
    }

    /**
     * Get user's progress for all concepts or a specific one
     * @param {string|null} concept - Optional concept slug to filter
     * @returns {Promise<Array>}
     */
    async getProgress(concept = null) {
        if (this.useStatic) {
            // Return cached progress from localStorage
            return this._getLocalProgress();
        }

        try {
            const endpoint = concept ? `/api/progress?concept=${concept}` : '/api/progress';
            const data = await this.fetchJSON(endpoint);
            return data.progress || [];
        } catch (e) {
            console.error('Failed to get progress:', e);
            return this._getLocalProgress();
        }
    }

    _getLocalProgress() {
        try {
            const key = `learning_framework_progress_${UserManager.getUserId()}`;
            return JSON.parse(localStorage.getItem(key)) || [];
        } catch {
            return [];
        }
    }

    _saveLocalProgress(progress) {
        const key = `learning_framework_progress_${UserManager.getUserId()}`;
        localStorage.setItem(key, JSON.stringify(progress));
    }

    /**
     * Get concepts due for review (spaced repetition)
     * @returns {Promise<Array>}
     */
    async getDueReviews() {
        if (this.useStatic) {
            return [];
        }

        try {
            const data = await this.fetchJSON('/api/reviews');
            return data.reviews || [];
        } catch (e) {
            console.error('Failed to get reviews:', e);
            return [];
        }
    }

    /**
     * Record a review result
     * @param {string} concept - Concept slug
     * @param {number} quality - Quality of recall (0-5, where 5 is perfect)
     * @returns {Promise<{success: boolean}>}
     */
    async recordReview(concept, quality) {
        if (this.useStatic) {
            return { success: false };
        }

        try {
            return await this.fetchJSON('/api/reviews', {
                method: 'POST',
                body: JSON.stringify({ concept, quality })
            });
        } catch (e) {
            console.error('Failed to record review:', e);
            return { success: false, error: e.message };
        }
    }

    /**
     * Check backend health
     * @returns {Promise<{status: string, db_available: boolean}>}
     */
    async checkHealth() {
        try {
            return await this.fetchJSON('/health');
        } catch (e) {
            return { status: 'unavailable', db_available: false };
        }
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

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API, WebSocketManager, UserManager, ProgressQueue };
}
