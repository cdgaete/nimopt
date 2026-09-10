---
title: Files
description: The file a definition or a model writes, what each key carries, the two data sources, and the verbs that read and write it.
---

# Files

## `load`, `loads`, `save`

| Verb | Does |
| --- | --- |
| `load(path, data=None)` | reads a file; a `Definition`, or a `Model` where the file carries data or `data=` gives it |
| `loads(text, data=None)` | the same over text; a sidecar name in text is refused, because text has no directory |
| `save(what, path, inline=False, instructions=False)` | writes a definition's file, or a model's with an `.npz` beside it, or one file with an inline block when `inline=True`; `instructions=True` writes the comment block that describes the format at the top of the file |

`data=` is the mapping `build` takes or the path of an `.npz`. A file that
carries data and a `data=` together is refused.

`Definition.to_yaml(instructions=False)` and `Model.to_yaml(inline=False,
instructions=False)` return the text `save` writes, without a sidecar line:
only `save` writes a sidecar and the line that names it.

With `instructions=True`, every writer puts a fixed comment block at the top
of the text. The block describes the format: the keys and their order, the
defaults, the rules that decide which rows a constraint has, and the
expression syntax. It is the same text in every file and describes the
format rather than the model, so a reader given one file can interpret it
without this package. The block is a YAML comment: a file with it and a
file without it load to the same model.

## The file

| Key | Carries |
| --- | --- |
| `version` | `2`; any other value is refused naming the one this reader understands |
| `name`, `sense` | the model's |
| `sets` | a list of names |
| `aliases` | each alias to the set it names; absent where the model declares none |
| `parameters` | each name to its dimensions |
| `variables` | each name to `sets`, and to `subset`, `lower`, `upper`, `integer` where they differ from no subset, `0`, infinity and `false` |
| `constraints` | each name to `relation`, and to `where` or `over` where given |
| `objective` | the expression's spelling; absent where the model states none |
| `data` | an inline mapping, or the name of an `.npz` beside the file |

`subset`, `where` and `over` take a parameter's name, meaning the coordinates
it carries, or a list of set names, meaning their full product. A symbol's
name is a Python identifier other than `Sum`. A dimension named in
`parameters`, in a variable's `sets`, or in a list of set names is a set or
an alias; an alias is declared after the set it names.

The expressions are spelled as they are typed in Python and read back through
the same operators, so the file is a fixed point: reading it and writing it
again gives the same text.

```python
from nimopt import Definition, Sum, loads

d = Definition("d")
S = d.set("S")
c = d.param("c", (S,))
x = d.var("x", (S,), integer=True)
d.eq("cap", 2 * c[S] * x[S] - 1 <= 5)
text = d.to_yaml()
print(text)
print(loads(text).to_yaml() == text)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
version: 2
name: d
sense: min
sets: [S]
parameters:
  c: [S]
variables:
  x:
    sets: [S]
    integer: true
constraints:
  cap:
    relation: (c[S] * 2) * x[S] - 1 <= 5

True
```

</details>
<!-- /output -->

A key the format does not carry is refused, at the top level and inside an
entry.

```python raises=ValueError
from nimopt import loads

loads(
    "version: 2\nname: d\nsense: min\nsets: [S]\n"
    "variables:\n  x: {sets: [S], bound: 1}\n"
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: variable 'x' carries ['bound'], which the format does not; it takes ('sets', 'subset', 'lower', 'upper', 'integer')
```

</details>
<!-- /output -->

## Data

The data a file carries is the mapping `build` takes, in three shapes.

| Shape | Inline | In the `.npz` |
| --- | --- | --- |
| a set's members | a list | a one-dimensional label array |
| a dense parameter | nested lists in row-major order | its grid |
| a long parameter | `columns`, the dimensions then `value`, and `rows` | a structured array with one field per dimension and `value` |

A parameter is written dense where its array covers its full product and
long otherwise. The `.npz` is read with `allow_pickle=False`; an array of
object dtype is refused at save, naming the symbol, because the container
would pickle it silently and the reader would then refuse the file.

```python raises=ValueError
from nimopt import loads

loads(
    "version: 2\nname: d\nsense: min\nsets: [S]\nparameters:\n  c: [S]\n"
    "data:\n  S: [a]\n  c:\n    columns: [value, S]\n    rows:\n    - [1.0, a]\n"
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: parameter 'c' is given columns ['value', 'S']; a table states the dimensions then value: ['S', 'value']
```

</details>
<!-- /output -->

## What is refused before anything is written

| Written | Refused because |
| --- | --- |
| a model whose `subset`, `where` or `over` is a domain with no name | declare the members as a parameter |
| two parameter objects or two set objects sharing a name in one model | the file keys a symbol by name |
| a symbol whose name is not an identifier, or is `Sum` | the spelling cannot address it |
| an array of object dtype | the container would pickle it |

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, subset

P = Set("P", np.array(["a", "b"]))
m = Model("m")
x = m.var("x", (P,))
m.eq("cap", x[P] <= 1.0, where=subset((P,), {"P": np.array(["a"])}))
m.to_yaml()
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'cap' states where= with a domain that has no name; declare its members as a parameter and name that
```

</details>
<!-- /output -->
