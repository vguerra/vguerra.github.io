---
title: "Python Basics"
description: "`match`, walrus, reshape/transpose tricks, string ops, dict max, char↔int"
category: "NumPy & Python"
order: 9
updatedDate: "2026-07-08T19:08:49.579Z"
---
## Switch Statement (match)

Python has no `switch`. Use `match` (3.10+) or `if/elif`.

```python
# match — modern, supports pattern matching
match value:
    case 1:
        print("one")
    case 2:
        print("two")
    case _:       # default
        print("other")

# Pattern matching examples
match point:
    case (0, 0):
        print("origin")
    case (x, 0):
        print(f"on x-axis at {x}")
    case (x, y):
        print(f"at {x}, {y}")
```

```python
# if/elif — universal fallback
if value == 1:
    print("one")
elif value == 2:
    print("two")
else:
    print("other")
```

---

## Walrus Operator `:=`

Assign and test in one expression — avoids repeating the expression in the body:

```python
if (diff := target - num) in seen:
    return [seen[diff], idx]
```

Parentheses around the assignment are important for precedence — without them:
```python
if diff := target - num in seen:   # wrong: evaluates as target - (num in seen)
```

Common uses:
```python
while chunk := f.read(1024):       # read until empty
    process(chunk)

if m := re.match(pattern, s):      # match and use result
    print(m.group(0))
```

---

## Reshape in Pure Python

Flat list → rows of length k:
```python
[flat[i:i+k] for i in range(0, len(flat), k)]   # slicing — most readable

it = iter(flat)
list(zip(*[it]*k))    # iter/zip trick: k refs to the SAME iterator
```

Flatten (reverse):
```python
[x for row in matrix for x in row]

from itertools import chain
list(chain.from_iterable(matrix))
```

Both approaches are O(n).

---

## Transpose a List of Lists (zip trick)

```python
matrix = [[1, 2, 3],
          [4, 5, 6]]

list(zip(*matrix))                    # [(1, 4), (2, 5), (3, 6)]
[list(row) for row in zip(*matrix)]   # [[1, 4], [2, 5], [3, 6]]
```

`*matrix` unpacks rows as separate args to `zip`, which pairs same-index elements across rows — exactly the columns.

---

## String Case Conversion

```python
s.lower()   # "Hello" → "hello"
s.upper()   # "Hello" → "HELLO"
s.title()   # "hello world" → "Hello World"
```

---

## Check Alphanumeric

```python
"a".isalnum()      # True
"1".isalnum()      # True
"!".isalnum()      # False
"abc123".isalnum() # True — all chars must be alphanumeric
"abc 123".isalnum() # False — space fails

[c for c in s if c.isalnum()]  # keep only alphanumeric chars
```

Related: `isalpha()` (letters only), `isdigit()` (digits only), `isspace()` (whitespace only).

---

## String Substitution

```python
s.replace('a b', 'ab')          # simple substitution

import re
re.sub(r'a b', 'ab', s)         # regex substitution for complex patterns
```

---

## Max Key by Value in a Dict

```python
max(freq_dict, key=freq_dict.get)

# example
freq = {'a': 3, 'b': 7, 'c': 1}
max(freq, key=freq.get)   # 'b'
```

Ties: `max()` returns the **first one encountered** (insertion order, Python 3.7+).

To get **all** keys with the max value:
```python
top = max(freq.values())
[k for k, v in freq.items() if v == top]   # ['a', 'b']
```

---

## List of N Empty Lists

```python
[[] for _ in range(n)]   # correct — each is a distinct list
```

Avoid:
```python
[[]] * n   # wrong — all n point to the same list object
```

---

## Char ↔ Int Conversion

```python
ord('a')   # 97
ord('A')   # 65

chr(97)    # 'a'
```

---

## Check All Values in a List

```python
all(v == 0 for v in lst)   # True if every element is 0

# numpy
np.all(arr == 0)
```
