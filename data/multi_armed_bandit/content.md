---
name: Multi-Armed Bandit
topic: Reinforcement Learning
difficulty: Beginner
tags: [rl, exploration_exploitation, algorithms]
---

# 多臂老虎机 (Multi-Armed Bandit)

多臂老虎机问题是强化学习中最基础的问题之一，它剥离了“状态”的概念，专注于**探索 (Exploration)** 与 **利用 (Exploitation)** 的权衡。

## 问题定义

假设你面前有 $k$ 台老虎机（Bandit），每台机器的奖励概率分布不同且未知。你的目标是在有限的操作次数内，通过选择不同的机器，最大化累积奖励。

- **动作 (Action)**: 选择拉动哪一台机器的手柄。
- **奖励 (Reward)**: 机器吐出的硬币（数值）。
- **价值 (Value)**: 每个动作的期望奖励 $q_*(a) = \mathbb{E}[R_t | A_t = a]$。

## 解决方案

### 1. 估算价值 (Action-Value Methods)

既然我们不知道真实的奖励概率，就只能通过尝试来估计。最简单的方法是**样本平均 (Sample Average)**：

$$ Q_t(a) \doteq \frac{\text{动作 } a \text{ 获得的奖励总和}}{\text{动作 } a \text{ 被选择的次数}} $$

#### 增量式实现 (Incremental Implementation)

为了节省内存，我们不需要存储所有历史奖励。可以使用递推公式更新估值：

$$ Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n] $$

代码实现 (参考 `avg.py`)：
```python
# Q: 当前估值, reward: 新获得的奖励, n: 该动作被选择的次数
Q = Q + (reward - Q) / n
```

### 2. 动作选择策略 (\epsilon-Greedy)

- **Greedy (贪婪)**: 总是选择当前估值最高 ($Q$ 最大) 的动作。这叫“利用”。
- **\epsilon-Greedy**:
    - 以概率 $1-\epsilon$ 选择贪婪动作（利用）。
    - 以概率 $\epsilon$ 随机选择一个动作（探索）。

这确保了所有动作都有机会被选中，从而随着时间推移收敛到真实值。

```