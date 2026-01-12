"""Visualizations for Loss Functions concept"""

import numpy as np
import matplotlib.pyplot as plt


def main_visualization():
    """Compare common loss functions visually
    
    Shows MSE, MAE, and binary cross-entropy loss surfaces.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE (Mean Squared Error)
    ax = axes[0, 0]
    y_true = 0
    y_pred = np.linspace(-3, 3, 100)
    mse = (y_pred - y_true)**2
    
    ax.plot(y_pred, mse, 'b-', linewidth=2, label='MSE')
    ax.scatter([y_true], [0], color='red', s=100, marker='*', zorder=5,
              label='Target (y=0)')
    ax.set_xlabel('Prediction ŷ')
    ax.set_ylabel('Loss')
    ax.set_title('Mean Squared Error: (ŷ - y)²', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE (Mean Absolute Error)
    ax = axes[0, 1]
    mae = np.abs(y_pred - y_true)
    
    ax.plot(y_pred, mae, 'g-', linewidth=2, label='MAE')
    ax.scatter([y_true], [0], color='red', s=100, marker='*', zorder=5,
              label='Target (y=0)')
    ax.set_xlabel('Prediction ŷ')
    ax.set_ylabel('Loss')
    ax.set_title('Mean Absolute Error: |ŷ - y|', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Binary Cross-Entropy
    ax = axes[1, 0]
    p = np.linspace(0.01, 0.99, 100)  # Predicted probability
    
    # y=1 case
    bce_y1 = -np.log(p)
    ax.plot(p, bce_y1, 'r-', linewidth=2, label='y=1 (true class)')
    
    # y=0 case
    bce_y0 = -np.log(1 - p)
    ax.plot(p, bce_y0, 'b-', linewidth=2, label='y=0 (false class)')
    
    ax.set_xlabel('Predicted Probability ŷ')
    ax.set_ylabel('Loss')
    ax.set_title('Binary Cross-Entropy: -y log(ŷ) - (1-y)log(1-ŷ)',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 5])
    
    # MSE vs MAE comparison
    ax = axes[1, 1]
    errors = np.linspace(0, 3, 100)
    mse_errors = errors**2
    mae_errors = errors
    
    ax.plot(errors, mse_errors, 'b-', linewidth=2, label='MSE')
    ax.plot(errors, mae_errors, 'g-', linewidth=2, label='MAE')
    ax.set_xlabel('Prediction Error |ŷ - y|')
    ax.set_ylabel('Loss Contribution')
    ax.set_title('MSE vs MAE: Effect of Outliers', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate('MSE penalizes\nlarge errors more!',
               xy=(2.5, 6.25), xytext=(1.5, 7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig


def loss_convergence():
    """Simulated loss convergence during training
    
    Shows typical training loss curve over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate training loss
    epochs = np.arange(0, 100)
    
    # Initial high loss, exponential decay to plateau
    train_loss = 2.5 * np.exp(-epochs / 20) + 0.1 + 0.05 * np.random.randn(100).cumsum() * 0.01
    val_loss = 2.5 * np.exp(-epochs / 20) + 0.15 + 0.05 * np.random.randn(100).cumsum() * 0.01
    
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3])
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
