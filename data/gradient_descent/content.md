---
name: Gradient Descent
topic: Optimization
difficulty: Beginner
tags: [optimization, fundamentals, algorithms]
---

# 梯度下降 (Gradient Descent)

梯度下降是一种迭代优化算法，用于寻找函数的局部最小值。它是神经网络训练的**基石**。

## 核心原理

算法的核心思想是沿着函数梯度的**反方向**移动。

### 数学表达

更新规则如下：
$$ \theta = \theta - \alpha \cdot \nabla J(\theta) $$

其中：
- $\theta$: 待优化的参数
- $\alpha$: 学习率 (Learning Rate)
- $\nabla J(\theta)$: 损失函数关于参数的梯度

## Python 伪代码

```python
def gradient_descent(params, learning_rate, gradient):
    return params - learning_rate * gradient
```

> [!TIP]
> 学习率 $\alpha$ 的选择至关重要：太大可能导致震荡不收敛，太小则会导致训练极慢。

```
