/**
 * Main application controller
 */
class App {
    constructor() {
        this.currentConcept = null;
        this.currentView = 'concepts';
        this.concepts = [];

        // Initialize
        this.init();
    }

    async init() {
        // Load concepts
        await this.loadConcepts();

        // Setup event listeners
        this.setupNavigation();
        this.setupConceptList();

        // Try WebSocket connection (optional)
        try {
            wsManager.connect();
            wsManager.on('training_update', (data) => this.handleTrainingUpdate(data));
        } catch (e) {
            console.log('WebSocket not available (training visualization disabled)');
        }

        // Update status
        this.setStatus('Ready');
    }

    async loadConcepts() {
        try {
            this.concepts = await api.getConcepts();
            this.renderConceptList();
        } catch (e) {
            console.error('Failed to load concepts:', e);
            this.setStatus('Error loading concepts');
        }
    }

    renderConceptList() {
        const listEl = document.getElementById('concept-list');
        listEl.innerHTML = '';

        this.concepts.forEach(concept => {
            const li = document.createElement('li');
            li.dataset.slug = concept.slug;
            li.innerHTML = `
                <span class="name">${concept.name}</span>
                <span class="topic">${concept.topic}</span>
            `;
            li.addEventListener('click', () => this.selectConcept(concept.slug));
            listEl.appendChild(li);
        });
    }

    setupNavigation() {
        document.querySelectorAll('.nav a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = e.target.dataset.view;
                this.switchView(view);
            });
        });
    }

    setupConceptList() {
        // Already handled in renderConceptList
    }

    switchView(view) {
        this.currentView = view;

        // Update nav
        document.querySelectorAll('.nav a').forEach(a => {
            a.classList.toggle('active', a.dataset.view === view);
        });

        // Update content
        if (view === 'graph') {
            this.showGraphViewer();
        } else if (view === 'training') {
            this.showTrainingView();
        } else {
            this.showConceptsView();
        }
    }

    async selectConcept(slug) {
        this.currentConcept = slug;

        // Update list selection
        document.querySelectorAll('.concept-list li').forEach(li => {
            li.classList.toggle('active', li.dataset.slug === slug);
        });

        // Load concept detail
        try {
            const concept = await api.getConceptDetail(slug);
            document.getElementById('viz-title').textContent = concept.name;

            // Load visualization
            await this.loadVisualization(slug);
        } catch (e) {
            console.error('Failed to load concept:', e);
            this.setStatus(`Error loading ${slug}`);
        }
    }

    async loadVisualization(concept, vizName = 'main_visualization') {
        try {
            const data = await api.getVisualizationData(concept, vizName);

            // Render based on visualization type
            if (data.type === 'graph') {
                graphRenderer.render(data);
            } else if (data.type === 'training') {
                trainingRenderer.render(data);
            } else {
                canvasRenderer.render(data);
            }

            this.setStatus(`Loaded: ${concept}`);
        } catch (e) {
            console.error('Failed to load visualization:', e);
            this.setStatus('Error loading visualization');
        }
    }

    showGraphViewer() {
        document.getElementById('viz-title').textContent = 'Computational Graph Viewer';
        // Graph-specific UI updates
    }

    showTrainingView() {
        document.getElementById('viz-title').textContent = 'Real-time Training';
        // Training-specific UI updates
    }

    showConceptsView() {
        if (this.currentConcept) {
            this.selectConcept(this.currentConcept);
        } else {
            document.getElementById('viz-title').textContent = 'Select a concept';
        }
    }

    handleTrainingUpdate(data) {
        if (this.currentView === 'training') {
            trainingRenderer.update(data);
        }
    }

    setStatus(message) {
        const statusEl = document.querySelector('.status-text');
        if (statusEl) {
            statusEl.textContent = message;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
