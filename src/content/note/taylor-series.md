---
title: "Taylor Series"
description: "definition, common expansions, computation, relevance to ML"
category: "Math Foundations"
order: 28
updatedDate: "2026-07-05T12:13:51.020Z"
---
## Definition

Approximates a function `f(x)` around a point `a` using its derivatives:

```
f(x) = f(a) + f'(a)(x-a) + f''(a)/2! (x-a)² + f'''(a)/3! (x-a)³ + ...
```

More terms → better approximation. Expanded around `a=0` it's called a **Maclaurin series**.

## Common Examples

```
eˣ     = 1 + x + x²/2! + x³/3! + ...
sin(x) = x - x³/3! + x⁵/5! - ...
cos(x) = 1 - x²/2! + x⁴/4! - ...
```

## NumPy — Numerical Evaluation

```python
import numpy as np

x = np.linspace(-3, 3, 100)

# Approximate eˣ around 0 up to degree 4
taylor_exp = sum(x**n / np.math.factorial(n) for n in range(5))

# Compare to true value
true_exp = np.exp(x)
```

## SymPy — Symbolic Expansion

```python
from sympy import symbols, series, exp, sin

x = symbols('x')
series(exp(x), x, 0, n=5)   # eˣ up to 5 terms around x=0
# 1 + x + x²/2 + x³/6 + x⁴/24 + O(x⁵)

series(sin(x), x, 0, n=6)
```

Use SymPy for symbolic expansions, NumPy for numerical evaluation.

## Usage in ML

### 1. Gradient Descent — 1st order approximation
```
f(x) ≈ f(xₖ) + ∇f(xₖ)ᵀ(x - xₖ)
```
Minimizing this linear approximation gives the GD update.

### 2. Newton's Method — 2nd order approximation
```
f(x) ≈ f(xₖ) + ∇f(xₖ)ᵀ(x - xₖ) + ½(x - xₖ)ᵀH(x - xₖ)
```
Minimizing this gives `x ← x - H⁻¹∇f`.

### 3. Activation Function Analysis
Taylor expansions explain behavior near zero:
```
sigmoid(x) ≈ 0.5 + x/4    (1st order around 0)
tanh(x)    ≈ x - x³/3     (Maclaurin)
```
Useful for understanding vanishing gradients in saturated regions.

### 4. Learning Rate Convergence Bound
The condition `η < 2/L` comes from Taylor expanding the loss and bounding the error term.

### 5. Laplace Approximation (Bayesian DL)
Fits a Gaussian to a loss landscape using a 2nd order Taylor expansion around the MAP estimate.

**Core pattern:** every optimization method chooses how many Taylor terms to use — more terms = better steps but higher compute cost.
