# Tutorial: Understanding Backpropagation

This tutorial guides you through implementing backpropagation from scratch.

## Prerequisites

Before starting, ensure you understand:
- Basic calculus (derivatives, chain rule)
- Matrix multiplication
- What neural networks are

## Learning Path

### Tier 1: Conceptual Understanding

**1. Start the learning session:**
```bash
lf learn --concept backpropagation
```

**2. Take the quiz to assess your understanding:**
```bash
lf quiz --concept backpropagation
```

**Key concepts to master:**
- Forward pass: computing outputs from inputs
- Loss function: measuring prediction error
- Backward pass: computing gradients
- Chain rule: propagating gradients through layers

### Tier 2: Implementation

**1. View the challenge:**
```bash
lf challenge backprop_fill
```

This opens a fill-in-the-blank challenge where you implement:
- Forward pass through a layer
- Backward pass (gradient computation)
- Weight updates

**2. Implement your solution:**

Edit the file at `user_data/implementations/backprop_fill.py`

**Key implementation steps:**

```python
# Forward pass: output = activation(input @ weights + bias)
def forward(self, x):
    self.x = x  # Cache for backward
    self.z = x @ self.W + self.b
    self.a = sigmoid(self.z)
    return self.a

# Backward pass: compute gradients
def backward(self, grad_output):
    # Gradient through activation
    grad_z = grad_output * sigmoid_derivative(self.z)

    # Gradient w.r.t. weights: dL/dW = x.T @ grad_z
    self.grad_W = self.x.T @ grad_z

    # Gradient w.r.t. bias: dL/db = sum(grad_z)
    self.grad_b = np.sum(grad_z, axis=0)

    # Gradient to pass to previous layer: dL/dx = grad_z @ W.T
    grad_input = grad_z @ self.W.T
    return grad_input
```

**3. Test your implementation:**
```bash
lf test backprop_fill
```

This runs:
- Unit tests comparing your output to reference
- Numerical gradient checking

**4. Debug if needed:**

If gradient check fails:
- Review the chain rule application
- Check matrix dimensions (should match)
- Verify you're caching values in forward pass

### Tier 3: Visualization

**1. Launch the visualization:**
```bash
lf viz backpropagation
```

**2. Explore interactively:**
- Watch gradients flow backward through the network
- See how weight updates affect outputs
- Experiment with different learning rates

### Tier 4: Scale (Optional)

**1. Estimate GPU cost:**
```bash
lf scale backprop_fill.py --estimate
```

**2. Submit for GPU training:**
```bash
lf scale backprop_fill.py --dataset mnist --epochs 30
```

**3. Monitor progress:**
```bash
lf status <job_id>
```

## Key Insights

### Why Backpropagation Works

Backpropagation efficiently computes gradients using the chain rule:
- Forward pass: compute and cache intermediate values
- Backward pass: use chain rule to compute gradients
- Key insight: reuse cached values to avoid redundant computation

### Common Mistakes

1. **Forgetting to cache values**: Need forward values in backward pass
2. **Wrong dimensions**: Matrix multiply order matters
3. **Missing gradient terms**: Every path contributes to gradient
4. **Not handling batches**: Average gradients over batch

### Performance Tips

- Use matrix operations (not loops) for efficiency
- Numerical stability: clip gradients, use log-sum-exp for softmax
- Memory efficiency: don't store unnecessary intermediates

## Next Steps

After mastering backpropagation:
1. Try SGD from scratch: `lf challenge sgd_scratch`
2. Learn about Adam optimizer
3. Explore convolutional backpropagation

## Reference

- [Backprop visualization](../PHASE4_USAGE.md#backpropagation-visualization)
- [Gradient checking](../PHASE3_USAGE.md#gradient-checking)
- [Python reference implementation](../../data/baselines/backprop_reference.py)
