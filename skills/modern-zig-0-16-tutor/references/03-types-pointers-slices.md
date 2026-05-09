# 03 — Types, pointers, slices (0.16.0)

The single biggest source of bugs in Zig. Memorize the shapes below before
you write any nontrivial code.

## Pointer flavors

| Type | Meaning | Operations |
|---|---|---|
| `*T` | Single-item pointer | `.*` deref, `[0..1]` → `*[1]T`, address compare |
| `*const T` | Read-only single-item pointer | same, no assignment through `.*` |
| `[*]T` | Many-item pointer | `p[i]`, `p[a..b]`, `p + n`, `p - n`; `T` must be sized |
| `[*:s]T` | Sentinel-terminated many-item pointer | same + guaranteed element at `len` == `s` |
| `[*c]T` | C pointer; allows 0, coerces to int | Only accept from translate-c output; never construct by hand |
| `[]T` | Slice = `{ ptr: [*]T, len: usize }` | `s[i]`, `s[a..b]`, `s.len`, `s.ptr` |
| `[:s]T` | Sentinel-terminated slice | same + `s[s.len] == s` |
| `*[N]T` | Pointer to a known-length array | `.len`, coerces to `[]T` and `[*]T` |
| `?*T` | Optional pointer; null-optimized | Has same size as `*T`; use in place of `*allowzero T` |

## `&x` takes an address

`&local` has type `*@TypeOf(local)` with const-ness inherited from the
binding. Returning `&local` from a function is a compile error in 0.16.

## Builtin renames (memorize)

| Stale | Modern 0.16 |
|---|---|
| `@intToPtr(*T, addr)` | `@ptrFromInt(addr)` (target type from result location) |
| `@ptrToInt(p)` | `@intFromPtr(p)` |
| `@ptrCast(*T, p)` (two-arg) | `@ptrCast(p)` (target from result location) |

```zig
// CORRECT
const ptr: *i32 = @ptrFromInt(0xdeadbee0);
const addr: usize = @intFromPtr(ptr);
const words: *const u32 = @ptrCast(&bytes);
```

## Explicitly-aligned pointers are now distinct types

`*u8` and `*align(1) u8` are no longer the same type. They still coerce
interchangeably (in-memory coercion), but `@TypeOf(a) == @TypeOf(b)` and
`@typeInfo(...)` comparisons see them as distinct. Relevant mainly in
generic helpers and reflection code.

## Slices — the fat pointer

```zig
var arr = [_]i32{ 10, 20, 30, 40 };

// Slicing with runtime bounds → []i32
var i: usize = 0; _ = &i;        // force runtime-known
const s: []i32 = arr[i..arr.len];

// Slicing with comptime bounds → *[N]T
const first_two: *[2]i32 = arr[0..2];

// The "slice by length" trick: runtime start, comptime length
const n = 2;
const window: *[n]i32 = arr[i..][0..n];

// Empty slice
const empty: []const u8 = &.{};
```

**Bounds are checked in Debug / ReleaseSafe.** `slice[slice.len]` traps;
`slice[slice.len]` on a sentinel-terminated slice `[:0]u8` is **allowed** and
equals the sentinel.

## Sentinel-terminated

String literals coerce to both `[]const u8` and `[:0]const u8`. When calling
C APIs, pass `[*:0]const u8`:

```zig
pub extern "c" fn puts(s: [*:0]const u8) c_int;

pub fn main(init: std.process.Init) !void {
    _ = puts("hello"); // literal coerces
    _ = init;
}
```

## Ownership rules (mentor note)

- `[]T` does **not** own its memory. Someone must track the `Allocator` that
  produced it and call `allocator.free(slice)`.
- A function that returns `![]u8` **transfers ownership** to the caller.
  Document this in the doc comment.
- A function that takes `[]const u8` **borrows**. It must not retain the
  slice past the call.
- If in doubt, accept `allocator: std.mem.Allocator` and return
  owned memory. Readability >> cleverness.

## `@ptrCast` uses result-location inference

```zig
// WRONG (0.10-era)
const u32_ptr = @ptrCast(*const u32, &bytes);

// CORRECT
const u32_ptr: *const u32 = @ptrCast(&bytes);
```

If you can't give a clean result location, use `std.mem.bytesAsSlice` or
`@bitCast` instead. Avoid `[*c]T` except when reading translate-c output.

## `*volatile T`

Only for MMIO / hardware registers. Has nothing to do with concurrency —
for that, use `@atomicLoad`, `@atomicStore`, or `std.atomic`.

## Common mentor diagnostic questions

- "Who owns this slice? Which allocator will free it?"
- "Is this a borrow or a transfer?"
- "Is the length runtime-known or comptime-known? The type should reflect it."
- "Do you want `*[N]T` (a pointer-to-array) or `[]T` (a slice)?"

<!-- ~2.2k tokens · verified against Zig 0.16.0 language reference -->
