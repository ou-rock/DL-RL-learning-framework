"""Visualizations for Backpropagation concept"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def main_visualization():
    """Computational graph showing forward and backward passes

    Demonstrates how information flows forward and gradients flow backward.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define node positions
    nodes = {
        'x': (1, 4),
        'w1': (1, 2),
        'z1': (3, 3),
        'σ': (5, 3),
        'a1': (7, 3),
        'w2': (7, 1),
        'z2': (9, 2),
        'ŷ': (11, 2),
        'y': (11, 4),
        'L': (13, 3)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        circle = Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12,
                fontweight='bold', zorder=4)

    # Forward pass arrows (green, solid)
    forward_edges = [
        ('x', 'z1', 'multiply'),
        ('w1', 'z1', 'multiply'),
        ('z1', 'σ', 'sigmoid'),
        ('σ', 'a1', ''),
        ('a1', 'z2', 'multiply'),
        ('w2', 'z2', 'multiply'),
        ('z2', 'ŷ', ''),
        ('ŷ', 'L', 'loss'),
        ('y', 'L', 'target'),
    ]

    for src, dst, label in forward_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='green', alpha=0.7, zorder=2
        )
        ax.add_patch(arrow)

        # Add label
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, fontsize=8,
                   color='darkgreen', ha='center')

    # Backward pass arrows (red, dashed)
    backward_edges = [
        ('L', 'ŷ', '∂L/∂ŷ'),
        ('ŷ', 'z2', '∂L/∂z2'),
        ('z2', 'w2', '∂L/∂w2'),
        ('z2', 'a1', '∂L/∂a1'),
        ('a1', 'σ', '∂L/∂σ'),
        ('σ', 'z1', '∂L/∂z1'),
        ('z1', 'w1', '∂L/∂w1'),
    ]

    for src, dst, label in backward_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15, linewidth=1.5,
            color='red', alpha=0.6, linestyle='--', zorder=1
        )
        ax.add_patch(arrow)

        # Add gradient label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y - 0.3, label, fontsize=7,
               color='darkred', ha='center', style='italic')

    # Legend
    ax.text(7, 6.5, 'Forward Pass (green, solid)', color='green',
           fontsize=12, fontweight='bold')
    ax.text(7, 6, 'Backward Pass (red, dashed)', color='red',
           fontsize=12, fontweight='bold')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Backpropagation: Forward & Backward Passes',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def gradient_flow():
    """Gradient magnitude through layers (vanishing gradient demo)

    Compares gradient flow with sigmoid vs ReLU activation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = ['Output', 'Layer 4', 'Layer 3', 'Layer 2', 'Layer 1', 'Input']
    x = range(len(layers))

    # Sigmoid: gradients vanish exponentially
    sigmoid_grads = [1.0, 0.25, 0.0625, 0.0156, 0.0039, 0.00098]

    # ReLU: gradients stay relatively constant
    relu_grads = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7]

    # Plot
    ax.bar([i - 0.2 for i in x], sigmoid_grads, width=0.4,
           label='Sigmoid', color='orange', alpha=0.7)
    ax.bar([i + 0.2 for i in x], relu_grads, width=0.4,
           label='ReLU', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Through Network Layers\n(Vanishing Gradient Problem)',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('Sigmoid gradients\nvanish rapidly!',
               xy=(4, 0.01), xytext=(3, 0.2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
