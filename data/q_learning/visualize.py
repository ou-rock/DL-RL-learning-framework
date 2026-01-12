"""Visualizations for Q-Learning concept"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches


def main_visualization():
    """Q-Learning in grid world environment
    
    Shows Q-values and optimal policy in a simple grid world.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simple 5x5 grid world
    grid_size = 5
    
    # Define rewards: goal at (4,4), penalty at (2,2)
    rewards = np.zeros((grid_size, grid_size))
    rewards[4, 4] = 10  # Goal
    rewards[2, 2] = -5  # Trap
    
    # Simulate Q-values (simplified)
    # Higher Q-values near goal, lower near trap
    Q_values = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            dist_to_goal = abs(i - 4) + abs(j - 4)
            dist_to_trap = abs(i - 2) + abs(j - 2)
            Q_values[i, j] = 10 - 2*dist_to_goal - 3*max(0, 3-dist_to_trap)
    
    # Plot 1: Q-values heatmap
    ax1 = axes[0]
    im = ax1.imshow(Q_values, cmap='RdYlGn', interpolation='nearest')
    
    # Add grid
    for i in range(grid_size + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=1)
        ax1.axvline(i - 0.5, color='black', linewidth=1)
    
    # Annotate Q-values
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax1.text(j, i, f'{Q_values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Mark special states
    ax1.add_patch(Rectangle((3.5, 3.5), 1, 1, fill=False, edgecolor='gold', linewidth=4))
    ax1.text(4, 4.3, 'GOAL', ha='center', fontsize=9, fontweight='bold', color='gold')
    
    ax1.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor='red', linewidth=4))
    ax1.text(2, 2.3, 'TRAP', ha='center', fontsize=9, fontweight='bold', color='red')
    
    ax1.set_title('Q-Values in Grid World', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    plt.colorbar(im, ax=ax1, label='Q-value')
    
    # Plot 2: Optimal policy (arrows)
    ax2 = axes[1]
    ax2.imshow(Q_values, cmap='RdYlGn', alpha=0.3, interpolation='nearest')
    
    # Add grid
    for i in range(grid_size + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=1)
        ax2.axvline(i - 0.5, color='black', linewidth=1)
    
    # Draw policy arrows (greedy policy from Q-values)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == (4, 4):  # Goal
                continue
            if (i, j) == (2, 2):  # Trap
                continue
            
            # Determine best action (simplified)
            if j < 4 and Q_values[i, j+1] > Q_values[i, j]:
                dx, dy = 0.3, 0
            elif i < 4 and Q_values[i+1, j] > Q_values[i, j]:
                dx, dy = 0, 0.3
            elif j > 0:
                dx, dy = -0.3, 0
            else:
                dx, dy = 0, -0.3
            
            arrow = FancyArrowPatch((j, i), (j + dx, i + dy),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2, color='blue')
            ax2.add_patch(arrow)
    
    # Mark special states
    ax2.add_patch(Rectangle((3.5, 3.5), 1, 1, fill=False, edgecolor='gold', linewidth=4))
    ax2.text(4, 4.3, 'GOAL', ha='center', fontsize=9, fontweight='bold', color='gold')
    
    ax2.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor='red', linewidth=4))
    ax2.text(2, 2.3, 'TRAP', ha='center', fontsize=9, fontweight='bold', color='red')
    
    ax2.set_title('Optimal Policy π*(s) = argmax Q(s,a)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    
    plt.tight_layout()
    return fig


def q_learning_convergence():
    """Q-Learning convergence over episodes
    
    Shows how Q-values improve with training.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate Q-learning convergence
    episodes = np.arange(0, 500)
    
    # Initial Q-values are random/poor
    # Converge to optimal Q* over time
    np.random.seed(42)
    optimal_q = 8.5
    
    # Simulated Q-value for state-action pair
    q_values = optimal_q * (1 - np.exp(-episodes / 100)) + np.random.randn(500) * 0.5 * np.exp(-episodes / 80)
    
    # Moving average
    window = 20
    q_smooth = np.convolve(q_values, np.ones(window)/window, mode='valid')
    episodes_smooth = episodes[:len(q_smooth)]
    
    ax.plot(episodes, q_values, 'b-', alpha=0.3, linewidth=0.5, label='Q(s,a) per episode')
    ax.plot(episodes_smooth, q_smooth, 'r-', linewidth=2, label='Moving average')
    ax.axhline(y=optimal_q, color='g', linestyle='--', linewidth=2, label='Optimal Q*(s,a)')
    
    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel('Q-value', fontsize=12)
    ax.set_title('Q-Learning Convergence to Optimal Q*', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Q-values converge\nto optimal Q*',
               xy=(300, optimal_q), xytext=(200, 10),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=11, color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
