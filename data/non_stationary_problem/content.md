---
name: Non-Stationary Problem
topic: Reinforcement Learning
difficulty: Intermediate
tags: [rl, optimization, algorithms]
prerequisites: [multi_armed_bandit]
---

# 非平稳问题 (Non-Stationary Problem)

在标准的多臂老虎机问题中，每台机器的奖励概率是固定的（平稳的）。但在现实世界中，环境往往是**非平稳的 (Non-Stationary)**，即奖励分布会随时间改变。

## 样本平均的局限性

标准的样本平均法（Sample Average）给予所有历史数据相同的权重 ($1/n$)：
$$ Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i $$

随着 $n$ 增大，$1/n$ 变得非常小，新的奖励 $R_n$ 对 $Q$ 值的影响微乎其微。如果环境变了，Agent 很难“感知”到变化。

## 解决方案：固定步长 ($\alpha$)

为了让 Agent 更关注**最近**的奖励，我们使用一个固定的步长参数 $\alpha \in (0, 1]$ 来替代 $1/n$：

$$ Q_{n+1} = Q_n + \alpha [R_n - Q_n] $$

### 指数加权移动平均

这个更新公式实际上是一种**指数加权移动平均 (Exponential Recency-Weighted Average)**。展开公式可以看到：

$$ Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} R_i $$

距离现在越近的奖励 $R_i$，其权重 $(1-\alpha)^{n-i}$ 越大；越久远的奖励，权重呈指数级衰减。这使得 Agent 能够快速适应环境的变化。
