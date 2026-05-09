# 06 — Containers (0.16.0, unmanaged idiom)

In 0.16 the standard containers moved to the **unmanaged** style across the
board: the allocator is **not stored** inside the container; the caller
passes it to every mutating method.

> VERIFY WITH ZIG 0.16.0 LOCALLY whether the surviving spelling is `ArrayList(T)`
> (rename) or still `ArrayListUnmanaged(T)`. The release notes indicate a
> migration "toward one variant". Treat `ArrayList(T)` as the canonical name
> unless your compiler disagrees.

## `std.ArrayList` — canonical form

```zig
// WRONG
var list = std.ArrayList(u32).init(a);
defer list.deinit();
try list.append(1);
list.items[0] = 2;

// CORRECT (0.16)
var list: std.ArrayList(u32) = .empty;
defer list.deinit(a);
try list.append(a, 1);
list.items[0] = 2;
```

- `.empty` is the default-initialized form.
- `append`, `appendSlice`, `ensureTotalCapacity`, `ensureUnusedCapacity`,
  `insert`, `pop`, `resize`, `clone`, `toOwnedSlice`, `deinit` — all take
  `allocator` as the first runtime argument.

## `std.AutoHashMap` / `std.StringHashMap` / `std.AutoArrayHashMap`

```zig
// WRONG
var m = std.AutoHashMap([]const u8, u32).init(a);
defer m.deinit();
try m.put("x", 1);

// CORRECT
var m: std.StringHashMap(u32) = .empty;
defer m.deinit(a);
try m.put(a, "x", 1);
if (m.get("x")) |v| { _ = v; }
```

The managed variants (`StringArrayHashMap`, `AutoArrayHashMap`, etc.) were
**removed**. Use the renamed unmanaged forms:

| Old managed name | New unmanaged name |
|---|---|
| `AutoArrayHashMap` | `std.array_hash_map.Auto` |
| `StringArrayHashMap` | `std.array_hash_map.String` |
| `ArrayHashMap` | `std.array_hash_map.Custom` |

## `std.PriorityQueue` — rename table

```zig
// WRONG
var q = std.PriorityQueue(u32, void, lessThan).init(a, {});
try q.add(3);
_ = q.remove();

// CORRECT
var q: std.PriorityQueue(u32, void, lessThan) = .empty;
defer q.deinit(a);
try q.push(a, 3);
_ = q.pop();
```

| Stale method | Modern method |
|---|---|
| `add` | `push` |
| `addSlice` | `pushSlice` |
| `remove` | `pop` |
| `removeOrNull` | `pop` |
| `removeMin` | `popMin` |
| `removeMax` | `popMax` |
| `removeIndex` | `popIndex` |

`PriorityDequeue` has the same renames.

## Memory pools

`std.heap.MemoryPoolUnmanaged`, `MemoryPoolAlignedUnmanaged`, and
`MemoryPoolExtraUnmanaged` exist for arena-like fixed-size allocation. Pass
the allocator to `create` / `destroy`.

## Removed containers / relocated

- `std.SegmentedList` — removed.
- `std.Thread.Pool` — removed; use `std.Io.Group`.
- `fs.getAppDataDir` — removed.

## `MultiArrayList` for SoA layouts (ML-relevant)

For cache-friendly batches of structs, consider `std.MultiArrayList`:

```zig
var mal: std.MultiArrayList(struct { x: f32, y: f32, id: u32 }) = .{};
defer mal.deinit(a);
try mal.append(a, .{ .x = 1, .y = 2, .id = 42 });
// .items(.x) returns []f32 — columnar access
```

## Default-init literals

| Literal | Means |
|---|---|
| `.empty` | Empty, capacity 0, no allocation yet |
| `.{}` | Default struct literal (works when all fields have defaults) |

Use `.empty` for containers unless the docs say otherwise.

## Common review findings

- `var list = std.ArrayList(T).init(a);` — stale, flag.
- `list.append(v);` without allocator — stale, flag.
- `list.deinit();` without allocator — stale, flag.
- Freed with a different allocator than allocated — flag hard.
- `.{}` on an empty `ArrayList` — prefer `.empty` to signal intent.

## Common mentor diagnostic questions

- "Which allocator owns this list's storage?"
- "Does this container outlive the allocator? (If yes: big problem.)"
- "Would `MultiArrayList` be a better layout here for cache behavior?"

<!-- ~2.0k tokens · verified against Zig 0.16.0 release notes -->
