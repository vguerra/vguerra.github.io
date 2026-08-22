---
title: "Type Annotations in Python"
description: "what to annotate (signatures, not locals), modern syntax (`list[int]`, `X | None`), accept-broadly/return-specifically, dataclass/TypedDict/Literal/Protocol/Final, PEP 695 generics, pyright vs mypy (hints inert without a checker), `Any` vs `object`"
category: "NumPy & Python"
order: 14
updatedDate: "2026-08-21T12:15:17.131Z"
---
Short version: **annotate function signatures, let locals be inferred, use modern built-in syntax,
and run a static checker.** Annotations do nothing at runtime by default — an unchecked hint is just a
comment; a checker (pyright/mypy) is what turns it into a bug-catcher.

---

## 1. What to annotate

- **Always: function parameters and return types** — the contract at each boundary, where types earn
  their keep.
- **Usually skip: local variables** — the checker infers them. Annotate a local only when inference
  can't (e.g. `items: list[int] = []`).
- **Class attributes** — annotate them, ideally as `@dataclass` fields.

---

## 2. Use modern syntax (Python 3.10+)

```python
def f(xs: list[int], opts: dict[str, float]) -> str | None:
    ...
```

- **Built-in generics** `list`/`dict`/`tuple`/`set` — **not** `typing.List`/`Dict` (deprecated 3.9+).
- **`X | Y` unions** and **`X | None`** — **not** `Optional[X]`/`Union[X, Y]` (3.10+).
- **Never rely on implicit Optional** — write `x: int | None = None` explicitly.

---

## 3. Accept broadly, return specifically

- **Parameters:** abstract types from `collections.abc` — `Iterable`, `Sequence`, `Mapping` — so
  callers can pass any compatible container.
- **Returns:** concrete types (`list`, `dict`) so callers know exactly what they get.

```python
from collections.abc import Iterable
def total(xs: Iterable[int]) -> int: ...   # accepts list, tuple, generator…
```

---

## 4. Reach for the right tool

| Tool | Use for |
|---|---|
| **`@dataclass`** | records / structured data with named fields (beats bare dicts/tuples) |
| **`TypedDict`** | a dict you must keep, but want keys/value-types checked |
| **`Literal["a","b"]`** | a fixed set of string/int values (mode flags) |
| **`Enum`** | named constants |
| **`Protocol`** | structural / duck typing ("anything with `.read()`") — no inheritance needed |
| **`Final`** | constants that shouldn't be reassigned |
| **`Self`** (3.11+) | methods returning their own class |
| **`TypeAlias` / `type` stmt** | readable aliases (3.12: `type Vector = list[float]`) |

---

## 5. Generics (PEP 695, 3.12 syntax)

```python
def first[T](xs: Sequence[T]) -> T:      # no explicit TypeVar needed
    return xs[0]
```

(Pre-3.12: `T = TypeVar("T")` then `def first(xs: Sequence[T]) -> T:`.)

---

## 6. Run a static checker — non-negotiable

- **pyright** (powers VS Code's Pylance) — fast, strict, great inference. Good default.
- **mypy** — the original, widely used in CI.
- **Annotations are not enforced at runtime** without a separate tool (pydantic / beartype /
  typeguard). The static checker is what actually catches bugs — hints alone are inert.

---

## 7. Escape hatches — use sparingly

- **`Any`** *disables* checking for that value — avoid. Prefer **`object`** when you truly mean
  "anything" (it forces you to narrow before use).
- **`cast(T, x)`** and **`# type: ignore[code]`** — last resorts; always pin the specific error code.
- **`from __future__ import annotations`** — makes all annotations lazy strings (helps forward refs
  and import cycles); reasonable to add by default on 3.10+.

---

## Guiding principle

Types are **executable documentation that a checker verifies.** Annotate the *interfaces* precisely,
keep the *internals* light, and let pyright/mypy enforce. Over-annotating obvious locals is noise;
under-annotating public signatures is where real bugs hide.

**For ML code:** this pairs with **runtime shape-typing** ([[tensor-shape-typing]]) — standard hints
handle "is this a `Tensor` / `list[int]`," while jaxtyping adds "is it `(batch, seq, dim)`." Different
layers, both useful.

---

Related: [[tensor-shape-typing]], [[python-basics]], [[pytorch-basics]]
