---
title: "Python Generators — `yield`, `yield from`, and typing"
description: "`yield` (lazy produce-and-suspend, generator objects, infinite sequences) & `yield from` (delegate to sub-iterable, forward send/throw, capture inner `return` value), flatten-nested-strings gotcha, typing generators (`Iterator[T]` vs `Generator[Y,S,R]`)"
category: "NumPy & Python"
order: 14
updatedDate: "2026-08-24T17:02:53.271Z"
---
Generators produce values **lazily** (on demand), so they represent huge or infinite sequences with
O(1) memory instead of materializing a whole list.

---

## `yield` — the basics

`yield` turns a function into a **generator function**. Calling it doesn't run the body — it returns a
**generator object**; the body runs lazily, one piece at a time.

```python
def counter():
    print("start")
    yield 1
    yield 2
    yield 3

g = counter()          # nothing printed — body hasn't run yet
next(g)                # prints "start", returns 1 → SUSPENDS here
next(g)                # resumes after `yield 1`, returns 2
next(g)                # returns 3
next(g)                # body ends → raises StopIteration
```

**Mental model:** `yield` produces a value and **suspends** the function, freezing all local state
(variables, loop position). The next `next()` **resumes** exactly where it left off. That's what makes
generators lazy:

```python
def naturals():          # infinite — impossible as a list
    n = 0
    while True:
        yield n
        n += 1
```

A `for` loop is just repeated `next()` until `StopIteration`:
```python
for x in counter():      # next() under the hood, stops cleanly at StopIteration
    ...
```

**Two-way (advanced):** `yield` is also an *expression* — `value = yield x` sends `x` out and receives
whatever the caller passes via `g.send(...)`. Basis of coroutines; rarely needed for plain iteration.

---

## `yield from` — delegating to a sub-iterable

`yield from iterable` yields **every item** from that iterable. Naive expansion:

```python
yield from sub
# ≡
for item in sub:
    yield item
```

**Flattening / recursion** (treat `str` as an atomic leaf — see the string gotcha below):
```python
def flatten(x):
    for item in x:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)    # delegate to the recursive call
        else:
            yield item
```

**Chaining sequences:**
```python
def chain_all(*iterables):
    for it in iterables:
        yield from it            # concatenate several iterables into one stream
```

### `yield from` is more than a loop

It sets up a **transparent two-way channel** between the outer caller and the delegated inner
generator — the part a plain `for`-loop version can't do:

1. **Values** flow out from inner to caller.
2. **`send()` / `throw()`** from the caller are forwarded *into* the inner generator.
3. **The inner generator's `return` value** becomes the **value of the `yield from` expression**:

```python
def inner():
    yield 1
    yield 2
    return "done"            # captured, NOT yielded

def outer():
    result = yield from inner()   # result == "done" after inner finishes
    print(result)                 # "done"
    yield 3
```

Point 3 is the one people miss: inside a generator, `return value` doesn't yield `value` — it stashes
it in `StopIteration`, and **`yield from` is what lets you capture it**.

---

## The string gotcha (flattening nested strings)

A **`str` is itself iterable**, so a naive "recurse if iterable" flattener explodes `"abc"` into
`'a','b','c'`. Guard by recursing only into **`list`/`tuple`** (as above), *not*
`isinstance(item, Iterable)` — otherwise strings get shredded character-by-character.

---

## Typing a generator

The return type annotates **what it yields**, not the generator object (see [[type-annotations]]):

```python
from collections.abc import Iterator, Generator

def squares(n: int) -> Iterator[int]:         # common case: only yields
    for i in range(n):
        yield i * i

def gen() -> Generator[YieldT, SendT, ReturnT]:   # full form — only if you use send/return
    ...
```

- **`Iterator[T]`** for a plain generator — the right default.
- **`Generator[Y, S, R]`** only when you use `.send()` (S) or a meaningful `return` value (R).
  Unused slots are `None`: `Generator[int, None, None]` ≡ `Iterator[int]`.
- **Async generators** (`async def` + `yield`) → `AsyncIterator[T]` / `AsyncGenerator[Y, S]`.
- Use `collections.abc`, **not** the deprecated `typing.Iterator`/`Generator`.

---

## The distinction to state crisply

- **`yield x`** — produce *one* value, suspend, resume later.
- **`yield from iterable`** — produce *all* values from a sub-iterable, and (for generators)
  transparently forward `send`/`throw` and capture the sub-generator's `return` value.

Use `yield` to emit individual items; use `yield from` when **delegating to another iterable/generator**
(recursion, chaining, composition) — more readable and semantically richer than the manual `for` loop.

---

Related: [[type-annotations]], [[python-basics]]
