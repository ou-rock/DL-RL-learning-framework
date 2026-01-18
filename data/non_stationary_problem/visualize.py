"""Visualization for Non-Stationary Problem"""

import numpy as np
import matplotlib.pyplot as plt

class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        # Introduce non-stationarity: probabilities shift slightly every step
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        return 0

class Agent:
    """Standard Sample Average Agent"""
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

class AlphaAgent(Agent):
    """Agent with fixed step size alpha"""
    def __init__(self, epsilon, alpha, action_size=10):
        super().__init__(epsilon, action_size)
        self.alpha = alpha

    def update(self, action, reward):
        # Fixed step size update rule
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

def main_visualization():
    """Compare Sample Average vs Alpha Update in non-stationary env"""
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8
    
    agent_types = {
        'Sample Average (1/n)': lambda: Agent(epsilon),
        'Alpha Update (Fixed)': lambda: AlphaAgent(epsilon, alpha)
    }
    
    results = {}

    for name, agent_factory in agent_types.items():
        all_rates = np.zeros((runs, steps))

        for run in range(runs):
            agent = agent_factory()
            bandit = NonStatBandit()
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))

            all_rates[run] = rates

        results[name] = np.average(all_rates, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, avg_rates in results.items():
        ax.plot(avg_rates, label=name)
        
    ax.set_ylabel('Average Reward Rate')
    ax.set_xlabel('Steps')
    ax.set_title(f'Non-Stationary Environment: Adaptability Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
