# 13 — Style guide (0.16.0)

Zig's style is enforced mostly by `zig fmt`; the rest is community convention.
This is a tight subset. When in doubt, favor **explicitness**, **ownership
clarity**, and **test coverage** over cleverness.

## Formatting

- Run `zig fmt src/` before committing. `zig fmt --check` in CI.
- 4-space indentation, spaces not tabs.
- Braces on same line as `if`/`for`/`while`/`fn`.
- One declaration per line. One blank line between top-level decls.
- Trailing comma for multi-line argument lists.

## Naming

| Thing | Style | Example |
|---|---|---|
| File with only decls | `snake_case.zig` | `memory_pool.zig` |
| File exporting one public type | `TitleCase.zig` | `Tensor2D.zig` |
| Type / function returning `type` | `TitleCase` | `LinkedList(T)`, `Buffer` |
| Function | `camelCase` | `parseInt`, `readAll` |
| Variable / parameter | `snake_case` | `byte_count`, `row_stride` |
| Struct field | `snake_case` | `.row_count`, `.col_count` |
| Constant (top-level) | `snake_case` for values, `TitleCase` for types | `max_tokens`, `const Version = ...` |
| Error | `PascalCase` member name | `error.InvalidInput` |
| Namespace (module import) | `snake_case` | `const fs = std.fs;` |

## Size / count / offset / index (numeric naming)

| Suffix | Meaning | Example |
|---|---|---|
| `_count` | How many, no unit context | `rows_count` |
| `_len` | Length, generally in elements | `buf_len` |
| `_size` | Bytes, or element count if context is clear | `stride_size` |
| `_offset` | Byte or element offset | `row_offset` |
| `_index` | Specific index into an array | `last_index` |
| `_id` | Opaque identifier, not an index | `stream_id` |

Pick one and stick with it. Mixing `_count` and `_size` in the same struct is
a review finding.

## Allocator naming

| Name | Meaning |
|---|---|
| `gpa` | General-purpose allocator (long-lived) |
| `arena` | Scoped arena (request / frame / subsystem) |
| `scratch` | Short-lived scratch space, freed quickly |
| `a` | Informal inside a tight function |

When a function takes two allocators, always name them explicitly — never
`a` and `b`.

## `const` over `var`

- `const` wherever mutation is not needed.
- `var` triggers a review question: "is this mutation really required?"

## Small-to-large file layout

Inside a `.zig` file:

1. Imports.
2. Public types.
3. Public functions.
4. Private helpers.
5. Tests.

Anonymous `test { _ = @import("other.zig"); }` at the bottom to pull in
sub-file tests.

## Shape-check at API boundaries

For numerical code:

```zig
pub fn matmulNaive(a: Tensor2D, b: Tensor2D, out: *Tensor2D) !void {
    if (a.cols != b.rows) return error.ShapeMismatch;
    if (out.rows != a.rows or out.cols != b.cols) return error.OutShapeMismatch;
    // ...
}
```

Never silently broadcast. Never infer shapes from positional arguments.

## Comments

- **Doc comments** (`///`) on public items. Describe ownership, error set,
  invariants.
- **Regular comments** (`//`) for intent, not redescription of the code.
- Avoid `TODO` without a `TODO(owner):` prefix; otherwise the TODO is
  anonymous and immortal.

## Pure vs allocating functions

- Pure: takes `[]const T`, returns `T` or a struct; no allocator.
- Allocating: takes `allocator: std.mem.Allocator`, returns owned slice/ptr.
- Streaming: takes `io: std.Io` (or a `*Writer`), returns `!void`.

Never mix: don't make a function "sometimes allocates". Too hard to review.

## Red flags in reviews

- Function taking `anytype` with no constraints — usually needs a concrete
  interface.
- Function returning a slice with no documented owner.
- Manual `while (i < n) : (i += 1)` where `for (0..n) |i|` reads better.
- `catch {}` — always a bug or a hidden TODO.
- `.{}` on a container where `.empty` is idiomatic.
- `unreachable` on a path the caller can reach. Use an explicit error.

## Common mentor diagnostic questions

- "Is this name a count, a size, or a length?"
- "Is this function pure, allocating, or streaming?"
- "Would a failing test prove you fixed the bug?"
- "Would a shape-check at entry make the panic path impossible?"

<!-- ~2.0k tokens -->
