# 05 — Memory and allocators (0.16.0)

Zig has **no default allocator**. Every function that allocates takes an
`Allocator`. Ownership is explicit. Failure is always an error
(`error.OutOfMemory`).

## The allocator chooser (from the langref)

| Situation | Allocator |
|---|---|
| Writing a library | Accept `allocator: std.mem.Allocator` parameter |
| Linking libc, short app | `std.heap.c_allocator` |
| Fixed comptime-known size | `std.heap.FixedBufferAllocator` |
| CLI app: free everything on exit | `std.heap.ArenaAllocator.init(std.heap.page_allocator)` |
| Per-request / per-frame | Nested arena, maybe over FBA |
| Testing leak detection | `std.testing.allocator` |
| Testing OOM paths | `std.testing.FailingAllocator` |
| General-purpose Debug | `std.heap.DebugAllocator` |
| General-purpose ReleaseFast | `std.heap.smp_allocator` |

## Canonical `DebugAllocator` setup (0.16)

```zig
// In main, or wherever you own the allocator:
var gpa_state: std.heap.DebugAllocator(.{}) = .init;
defer {
    if (gpa_state.deinit() == .leak) {
        std.debug.print("leak detected\n", .{});
    }
}
const gpa = gpa_state.allocator();
```

**Mentor note.** This is the post-rename form. `GeneralPurposeAllocator(.{}){}`
is gone.

## `ArenaAllocator` is now thread-safe and lock-free

```zig
var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
defer arena.deinit();
const a = arena.allocator();

const buf = try a.alloc(u8, 1024);
// no free needed — arena frees in bulk on deinit
```

- Ideal for CLI apps, compilers, per-request handling.
- No more `std.heap.ThreadSafeAllocator` wrapper — it has been removed.

## Use `Init`-provided allocators in `main`

```zig
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;        // default GPA, leak-checks in Debug
    const arena = init.arena;    // arena that cleans up on exit
    _ = gpa; _ = arena;
}
```

## Alloc primitives

```zig
const slice = try a.alloc(T, n);       // []T, uninitialized
defer a.free(slice);

const one = try a.create(T);           // *T
defer a.destroy(one);

const z = try a.allocWithOptions(T, n, .@"align"(64), null); // align 64
defer a.free(z);

const grown = try a.realloc(old_slice, new_len);
```

## Ownership rules

- A function that **allocates and returns** a slice/pointer **transfers
  ownership**. Document it. Pair with `errdefer a.free(buf)` in the function.
- A function that **takes** a slice **borrows**. It must not retain the
  slice past the call.
- When in doubt, return `!T` that owns, or take `*std.ArrayList(T)` (also
  unmanaged style) so the caller controls lifetime.

## `errdefer` + alloc pattern

```zig
fn buildThing(a: std.mem.Allocator) !Thing {
    const buf = try a.alloc(u8, 1024);
    errdefer a.free(buf);

    const extra = try a.alloc(u32, 32);
    errdefer a.free(extra);

    try somePossiblyFailingInit(buf, extra);
    return Thing{ .buf = buf, .extra = extra };
}
```

Order matters: put each `errdefer` **immediately after** the matching
allocation, so an error from a later call unwinds the right allocations.

## `FixedBufferAllocator` — bounded, fast, zero-syscall

```zig
var stack_buf: [4096]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&stack_buf);
const a = fba.allocator();
const s = try a.alloc(u8, 256); // returns OOM if capacity exceeded
_ = s;
```

Useful for small scratch space in kernels, embedded code, or inside a
benchmark hot loop.

## `std.testing.allocator`

The test allocator **detects leaks** and **detects double-frees / bad
frees**. Always use it inside `test "..."`.

```zig
test "no leak" {
    const a = std.testing.allocator;
    const buf = try a.alloc(u8, 16);
    defer a.free(buf);
    // ...
}
```

If you want to test OOM paths, use `std.testing.FailingAllocator` wrapped
around the test allocator.

## Common review findings

- **`defer a.free(buf)` before the allocation could succeed.** The `defer`
  sees the old value — put it *after* the `try a.alloc(...)`.
- **No `errdefer` on intermediate allocations** in a constructor.
- **Library code calling a process-wide allocator** like `c_allocator` or
  `page_allocator` instead of accepting one. Red flag.
- **Hidden allocation** inside a "cheap" looking function.
- **Mixing allocators** — freeing a slice with a different allocator than
  allocated it. Undefined behavior.

## Common mentor diagnostic questions

- "Which allocator does this function use? Is it parameterized?"
- "If `try` fires on line N, what leaks?"
- "Is this a long-lived allocation (`gpa`) or short-lived (`arena`)?"
- "What's the upper bound on memory? Could you bound it with an FBA?"

<!-- ~2.2k tokens · verified against Zig 0.16.0 langref + release notes -->
