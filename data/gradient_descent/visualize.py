"""Visualizations for Gradient Descent concept"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main_visualization():
    """3D loss surface with gradient descent convergence path

    Shows how gradient descent navigates a convex loss surface
    to find the minimum.
    """
    fig = plt.figure(figsize=(14, 6))

    # Create 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Generate loss surface (simple quadratic bowl)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Convex function

    # Plot surface
    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

    # Gradient descent path (learning rate = 0.2)
    path_x = [4.0]
    path_y = [4.0]
    lr = 0.2

    for _ in range(10):
        x_curr = path_x[-1]
        y_curr = path_y[-1]
        # Gradient of x² + y² is (2x, 2y)
        grad_x = 2 * x_curr
        grad_y = 2 * y_curr
        # Update
        path_x.append(x_curr - lr * grad_x)
        path_y.append(y_curr - lr * grad_y)

    path_z = [x**2 + y**2 for x, y in zip(path_x, path_y)]

    ax1.plot(path_x, path_y, path_z, 'r-o', linewidth=2, markersize=6,
             label='GD Path (α=0.2)')

    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss J(θ)')
    ax1.set_title('Gradient Descent on 3D Loss Surface')
    ax1.legend()

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path_x, path_y, 'r-o', linewidth=2, markersize=6,
             label='GD Path')
    ax2.scatter([0], [0], color='red', s=100, marker='*',
                label='Minimum', zorder=5)
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('Contour View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def learning_rate_comparison():
    """Compare convergence with different learning rates

    Demonstrates effect of learning rate on convergence speed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    learning_rates = [0.01, 0.1, 0.5, 0.9]

    for ax, lr in zip(axes.flat, learning_rates):
        # Generate loss surface
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        # Gradient descent with this learning rate
        path_x = [4.0]
        path_y = [4.0]

        for _ in range(20):
            x_curr = path_x[-1]
            y_curr = path_y[-1]
            grad_x = 2 * x_curr
            grad_y = 2 * y_curr
            path_x.append(x_curr - lr * grad_x)
            path_y.append(y_curr - lr * grad_y)

            # Break if diverging
            if abs(path_x[-1]) > 10 or abs(path_y[-1]) > 10:
                break

        # Plot
        contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
        ax.plot(path_x, path_y, 'r-o', linewidth=1.5, markersize=4)
        ax.scatter([0], [0], color='red', s=100, marker='*', zorder=5)

        # Determine behavior
        if abs(path_x[-1]) > 10:
            behavior = "Diverges"
            color = 'red'
        elif len(path_x) < 20 and abs(path_x[-1]) < 0.01:
            behavior = "Fast convergence"
            color = 'green'
        else:
            behavior = "Slow convergence"
            color = 'orange'

        ax.set_title(f'α = {lr}: {behavior}', color=color, fontweight='bold')
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Test standalone
    fig = main_visualization()
    plt.show()
