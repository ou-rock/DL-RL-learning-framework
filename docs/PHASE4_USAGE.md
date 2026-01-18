# Phase 4: Interactive Visualization - Usage Guide

## Overview

Phase 4 adds interactive visualization capabilities:

1. **Web-based visualization server** - Local HTTP server with interactive UI
2. **Computational graph viewer** - Explore network architecture visually
3. **Parameter exploration** - Interactive sliders to explore hyperparameter effects
4. **Real-time training visualization** - Live training progress charts

## Getting Started

### Start Visualization Server

```bash
lf viz
```

This opens your browser at `http://localhost:8080` with the visualization interface.

Options:
- `--port 9000` - Use different port
- `--no-browser` - Don't auto-open browser

### View Static Visualizations

```bash
lf visualize gradient_descent
lf visualize backprop --viz gradient_flow
lf visualize loss_functions --output file
```

## Web Interface

### Concept Browser

The sidebar lists all available concepts. Click to:
- View computational graph
- Explore parameters
- See training curves

### Computational Graph Viewer

Interactive graph showing:
- **Blue nodes**: Hidden layers
- **Green nodes**: Input layer
- **Red nodes**: Output layer
- **Yellow nodes**: Operations (matmul, activation)
- **Solid arrows**: Forward pass
- **Dashed arrows**: Backward pass (gradients)

**Controls:**
- **Click + drag**: Pan view
- **Mouse wheel**: Zoom in/out
- **Click node**: Show details

### Parameter Exploration

Sliders for exploring hyperparameters:
- Learning rate (logarithmic scale)
- Batch size
- Momentum
- Custom parameters per concept

The loss landscape updates in real-time as you adjust parameters.

### Real-time Training

When training is active:
- Live loss curve
- Live accuracy curve
- Epoch progress
- Time estimates

## API Endpoints

The visualization server exposes REST API:

```
GET /health              - Server status
GET /api/concepts        - List concepts
GET /api/concept/{slug}  - Concept detail
GET /api/graph           - Computational graph data
GET /api/viz/{concept}/{viz}  - Visualization data
```

## Integration with Training

### Monitoring Training

```python
from learning_framework.visualization import TrainingMonitor

monitor = TrainingMonitor(total_epochs=100)

for epoch in range(100):
    loss = train_one_epoch()
    monitor.record(epoch=epoch, loss=loss, accuracy=accuracy)

    # Send to WebSocket for live updates
    ws.send(monitor.get_update_message())
```

### Parameter Exploration in Code

```python
from learning_framework.visualization import ParameterExplorer
import numpy as np

explorer = ParameterExplorer()

explorer.define_parameters({
    'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.01, 'scale': 'log'},
    'momentum': {'min': 0.0, 'max': 0.99, 'default': 0.9}
})

def loss_fn(params):
    # Your training logic
    return compute_validation_loss(params['learning_rate'], params['momentum'])

explorer.set_loss_function(loss_fn)

# Compute effect of learning rate
effect = explorer.compute_effect('learning_rate', num_points=50)
```

### Building Custom Graphs

```python
from learning_framework.visualization import ComputationalGraphBuilder

builder = ComputationalGraphBuilder()

# Feedforward network
graph = builder.build_feedforward_graph(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    output_dim=10,
    activations=['relu', 'relu', 'relu', 'softmax'],
    include_gradients=True
)

# From dezero variables
graph = builder.build_from_dezero(output_variables)
```

## Customization

### Adding New Visualizations

Create `data/{concept}/visualize.py`:

```python
import matplotlib.pyplot as plt
import numpy as np

def main_visualization():
    """Main visualization for this concept"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Your visualization code
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x))

    ax.set_title('My Visualization')
    return fig

def custom_viz():
    """Another visualization"""
    # ...
    return fig
```

### Custom CSS

Modify `learning_framework/visualization/web/css/main.css` for custom styling.

### Custom JavaScript

Add new renderers in `learning_framework/visualization/web/js/`.

## Troubleshooting

### Server won't start

1. Check port is available: `lsof -i :8080`
2. Try different port: `lf viz --port 9000`

### Visualizations not loading

1. Check concept exists in `data/` directory
2. Verify `visualize.py` has required functions
3. Check browser console for JavaScript errors

### WebSocket disconnections

1. Check firewall settings
2. WebSocket reconnects automatically (5 attempts)
3. Check server logs for errors

## Next Steps

- Phase 5: GPU Backend (Vast.ai, Colab)
- Phase 6: C++ Deep Implementations
- Phase 7: Polish & Documentation
