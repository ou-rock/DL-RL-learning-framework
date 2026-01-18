"""Visualization for Multi-Armed Bandit"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        return 0

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

def main_visualization():
    """Simulate Bandit and plot Average Reward over steps"""
    steps = 1000
    epsilon = 0.1
    runs = 200  # Average over multiple runs for smoother plot

    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_rates, label=f'Epsilon = {epsilon}')
    ax.set_ylabel('Average Reward (Rates)')
    ax.set_xlabel('Steps')
    ax.set_title(f'Multi-Armed Bandit Learning Curve (Avg of {runs} runs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
