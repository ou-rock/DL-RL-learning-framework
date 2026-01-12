"""Visualizations for Activation Functions concept"""

import numpy as np
import matplotlib.pyplot as plt


def main_visualization():
    """Compare common activation functions
    
    Shows sigmoid, tanh, ReLU, and Leaky ReLU side by side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.linspace(-5, 5, 200)
    
    # Sigmoid
    ax = axes[0, 0]
    sigmoid = 1 / (1 + np.exp(-x))
    ax.plot(x, sigmoid, 'b-', linewidth=2, label='σ(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('σ(x)')
    ax.set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(2, 0.2, 'Range: (0, 1)', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Tanh
    ax = axes[0, 1]
    tanh = np.tanh(x)
    ax.plot(x, tanh, 'g-', linewidth=2, label='tanh(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('tanh(x)')
    ax.set_title('Hyperbolic Tangent: tanh(x)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(2, -0.5, 'Range: (-1, 1)', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # ReLU
    ax = axes[1, 0]
    relu = np.maximum(0, x)
    ax.plot(x, relu, 'r-', linewidth=2, label='ReLU(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('ReLU(x)')
    ax.set_title('ReLU: max(0, x)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(2, 1, 'Range: [0, ∞)', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Leaky ReLU
    ax = axes[1, 1]
    alpha = 0.1
    leaky_relu = np.where(x > 0, x, alpha * x)
    ax.plot(x, leaky_relu, 'm-', linewidth=2, label=f'Leaky ReLU (α={alpha})')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('Leaky ReLU(x)')
    ax.set_title(f'Leaky ReLU: max(αx, x) where α={alpha}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(2, -0.3, 'Fixes dying ReLU', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def activation_derivatives():
    """Compare activation function derivatives
    
    Shows why ReLU avoids vanishing gradients.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.linspace(-5, 5, 200)
    
    # Sigmoid derivative
    ax = axes[0, 0]
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    ax.plot(x, sigmoid_grad, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel("σ'(x)")
    ax.set_title("Sigmoid Derivative: σ'(x) = σ(x)(1-σ(x))", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.3])
    ax.text(2, 0.25, 'Max gradient: 0.25\n(Vanishing!)', fontsize=10, color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Tanh derivative
    ax = axes[0, 1]
    tanh_grad = 1 - np.tanh(x)**2
    ax.plot(x, tanh_grad, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel("tanh'(x)")
    ax.set_title("Tanh Derivative: tanh'(x) = 1 - tanh²(x)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])
    ax.text(2, 0.9, 'Max gradient: 1.0\n(Better than sigmoid)', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # ReLU derivative
    ax = axes[1, 0]
    relu_grad = np.where(x > 0, 1, 0)
    ax.plot(x, relu_grad, 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel("ReLU'(x)")
    ax.set_title("ReLU Derivative: 1 if x>0, else 0", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.5])
    ax.text(2, 1.2, 'Constant gradient: 1\n(No vanishing!)', fontsize=10, color='green',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # All derivatives comparison
    ax = axes[1, 1]
    ax.plot(x, sigmoid_grad, 'b-', linewidth=2, label='Sigmoid', alpha=0.7)
    ax.plot(x, tanh_grad, 'g-', linewidth=2, label='Tanh', alpha=0.7)
    ax.plot(x, relu_grad, 'r-', linewidth=2, label='ReLU', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.set_title('Gradient Comparison', fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
