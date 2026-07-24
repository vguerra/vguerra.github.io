---
title: "PyTorch Basics"
description: "dtype conversion, `.item()`, clamp, round, NLL loss, sigmoid, sqrt, in-place updates & `no_grad`, `requires_grad`, `torch.autograd.grad`, one-hot, (n,)↔(n,1)"
category: "PyTorch — Tensors & Mechanics"
order: 0
updatedDate: "2026-07-13T14:29:13.202Z"
---
## In-Place Tensor Updates

```python
# Direct assignment
t[0] = 5              # single element
t[:, 0] = 10          # column
t[0:2] = 0            # slice

# In-place operations (trailing _ indicates in-place)
t.add_(5)             # t += 5
t.mul_(2)             # t *= 2
t.sub_(1)             # t -= 1
t.div_(2)             # t /= 2

# Copy from another tensor
t.copy_(other_tensor)

# Zero or fill
t.zero_()             # all zeros
t.fill_(5)            # all fives
```

**Caution:** in-place ops break the computation graph for backprop. When updating model parameters during training, use `torch.no_grad()`:
```python
with torch.no_grad():
    weights -= learning_rate * weights.grad
```

---

## Comparing Tensors

**Exact equality:**
```python
torch.equal(a, b)      # True/False — all elements match exactly
(a == b).all()         # same, but returns boolean tensor
```

**Floating point (with tolerance — preferred for ML):**
```python
torch.allclose(a, b)              # True/False
torch.allclose(a, b, atol=1e-4)   # custom tolerance
torch.isclose(a, b).all()         # element-wise, then check all
```

**Element-wise:**
```python
a == b              # boolean tensor (True/False per element)
(a == b).any()      # True if any match
(a == b).sum()      # count matches
```

**Best practice:** Always use `torch.allclose()` for floats — `==` fails due to precision. Never do `0.1 + 0.2 == 0.3` (False!).

---

## requires_grad — Enable/Disable Gradient Tracking

```python
t.requires_grad_(True)    # enable — in-place, note the underscore
t.requires_grad_(False)   # disable — in-place

t.requires_grad = True    # direct assignment also works
```

Create a new tensor without gradients:
```python
t_no_grad = t.detach()    # new tensor, shares storage, no grad tracking
```

Temporary disable (context):
```python
with torch.no_grad():
    # operations here don't track gradients
    y = model(x)
```

---

## Converting Tensor dtype

```python
tensor.int()              # float → int32
tensor.long()             # float → int64 (most common for indices)
tensor.to(torch.int32)    # explicit
```

`.long()` is most common — ops like `nn.Embedding` and loss functions expect `int64`.

---

## Extracting a Scalar from a Tensor

```python
tensor.item()    # single-element tensor → Python float
tensor.tolist()  # multi-element tensor → Python list
```

---

## torch.autograd.grad — Compute Gradients Explicitly

Computes gradients without calling `.backward()`, useful for higher-order derivatives and gradients w.r.t. intermediate values.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3*x + 1

# Compute dy/dx
grad_y = torch.autograd.grad(y, x)[0]   # returns tuple; [0] unpacks
# 7.0
```

**Multiple outputs:**
```python
y1 = x ** 2
y2 = x ** 3
grad_y1, grad_y2 = torch.autograd.grad((y1, y2), x)
```

**Higher-order derivatives (gradient of gradient):**
```python
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]  # create_graph=True for further differentiation
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
# 12.0 (second derivative of x^3 at x=2)
```

**Unused inputs:**
```python
torch.autograd.grad(y, (x, z), allow_unused=True)  # won't error if z unused in y
```

**With upstream gradient (chain rule):**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Upstream gradient (dL/dy)
upstream_grad = torch.tensor([3.0])

# Compute dL/dx = (dL/dy) * (dy/dx)
grad_x = torch.autograd.grad(y, x, grad_outputs=upstream_grad)[0]
# 3 * (2*x) = 12
```

Without `grad_outputs`, it defaults to `torch.ones_like(output)`.

**Multiple inputs:**
```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

z = x**2 + x*y + y**2

# Compute dz/dx and dz/dy
grad_x, grad_y = torch.autograd.grad(z, (x, y))
# (7.0, 8.0)

# With upstream gradient
grad_x, grad_y = torch.autograd.grad(z, (x, y), grad_outputs=torch.tensor([2.0]))
# (14.0, 16.0)
```

Pass a tuple of tensors → get back a tuple of gradients in the same order.

Key differences from `.backward()`:
- Returns gradient as tuple, doesn't modify `.grad` attribute
- Can compute w.r.t. multiple tensors simultaneously
- No need for `zero_grad()` in a loop

---

## Clipping Probabilities

```python
predicted_probs = torch.clamp(predicted_probs, min=1e-9, max=1.0)
```

Cleaner than adding epsilon directly — won't push values above 1.

---

## TypeError: Tensor doesn't define __round__

Python's built-in `round()` does not work on tensors.

```python
# Wrong:
round(tensor, 6)

# Correct:
torch.round(tensor, decimals=6)
tensor.round()          # nearest integer, keeps float dtype
tensor.round().long()   # round then convert to int64
```

---

## Negative Log-Likelihood Loss

```python
return -torch.mean(torch.log(predicted_probs))
```

Always clip before taking log to avoid `log(0) = -inf`:
```python
predicted_probs = torch.clamp(predicted_probs, min=1e-9, max=1.0)
return -torch.mean(torch.log(predicted_probs))
```

Select the true class probs from the full distribution first:
```python
predicted_probs = probs[torch.arange(len(labels)), labels]
```

**Built-in alternatives (numerically stable):**
```python
torch.nn.functional.cross_entropy(logits, labels)   # raw logits
torch.nn.functional.nll_loss(log_probs, labels)     # post log_softmax
```

---

## In-place Updates on Leaf Tensors (torch.no_grad)

Error: in-place ops (`-=`) on tensors with `requires_grad=True` fail — autograd tracks every op and an in-place update would corrupt the graph.

**Official pattern** — update inside `torch.no_grad()`:
```python
loss.backward()             # compute gradients

with torch.no_grad():       # update without tracking
    w -= lr * w.grad
    b -= lr * b.grad

w.grad.zero_()              # reset gradients after EVERY step (per batch, not per epoch)
b.grad.zero_()
```

**Step vs epoch:** epoch = full pass over dataset; step = one batch update. Zero gradients after every step — `backward()` *accumulates* into `.grad`, so skipping it adds the previous batch's gradients to the next. (In full-batch GD, step and epoch coincide.)

Alternatives:
- `w.data -= lr * w.grad` — works but discouraged (bypasses autograd safety checks)
- `torch.optim.SGD` — real-world solution; `optimizer.step()` does the no_grad update internally

---

## Tensor (i,) → Column Tensor (i,1)

```python
t.unsqueeze(1)       # PyTorch idiomatic — adds dim at index 1
t.reshape(-1, 1)     # portable (also numpy)
t[:, None]           # slicing trick
t.view(-1, 1)        # requires contiguous memory
```

Reverse: `t.squeeze(1)` — removes a dimension of size 1.

---

## Managing (n,) Tensors — Avoiding Broadcasting Bugs

**The classic trap** — mixing `(n,1)` and `(n,)`:
```python
F.mse_loss(pred, target)   # pred (n,1), target (n,) → broadcasts to (n,n) — WRONG
```

Best practices:
1. Match shapes explicitly at loss computation: `pred.squeeze(1)` or `target.unsqueeze(1)`
2. Assert shapes: `assert pred.shape == target.shape`
3. Convention: `(n,)` for labels/targets, `(n, d)` for features; squeeze model outputs when output dim is 1
4. Always pass explicit dim to `squeeze(1)` — bare `squeeze()` removes ALL size-1 dims (dangerous with batch size 1)

---

## Float Precision and Rounding

`torch.round(t, decimals=4)` rounds correctly, but values like `0.1234` can't be represented exactly in binary floating point — `.tolist()` shows the nearest representable float:

```python
torch.round(t, decimals=4).tolist()
# [0.12340000271797180]  ← nearest float32 to 0.1234
```

Options:
```python
[f"{x:.4f}" for x in t.tolist()]     # display — format as strings
[round(x, 4) for x in t.tolist()]    # round at Python level
torch.allclose(a, b, atol=1e-4)      # comparisons — use tolerance, never exact
```

Rounding a float gives the *closest representable* float, not the exact decimal.

---

## MSE Loss

```python
import torch.nn.functional as F

F.mse_loss(pred, target)                     # mean (default)
F.mse_loss(pred, target, reduction='sum')    # sum
F.mse_loss(pred, target, reduction='none')   # per-element, no reduction

# module form
criterion = torch.nn.MSELoss()
loss = criterion(pred, target)
```

Both tensors must have the same shape.

---

## Sigmoid

```python
torch.sigmoid(x)   # numerically stable, built-in
```

---

## Square Root

```python
math.sqrt(x)                   # plain Python scalar, not differentiable
torch.sqrt(torch.tensor(x))   # wrap scalar in tensor first
x ** 0.5                       # simpler, works on tensors and scalars
tensor.sqrt()                  # method form on existing tensor
```

Use `math.sqrt` for plain numbers outside the computation graph. Use `torch.sqrt` / `** 0.5` when inside a network (supports autograd).

---

## One-Hot Encoding

**Using F.one_hot (recommended):**
```python
import torch.nn.functional as F

# Single vector
class_idx = 2
n_classes = 5
one_hot = F.one_hot(torch.tensor(class_idx), num_classes=n_classes)
# tensor([0, 0, 1, 0, 0])

# Convert to float32
one_hot = F.one_hot(torch.tensor(class_idx), num_classes=n_classes).float()

# Batch
indices = torch.tensor([0, 2, 4, 1])
one_hot_batch = F.one_hot(indices, num_classes=n_classes)
# tensor([[1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0]])
```

**Manual approach:**
```python
one_hot = torch.zeros(n_classes)
one_hot[class_idx] = 1.0

# Batch
one_hot_batch = torch.zeros(len(indices), n_classes)
one_hot_batch[torch.arange(len(indices)), indices] = 1.0
```

`F.one_hot` returns `int64` by default — use `.float()` if you need float32 for computation.
