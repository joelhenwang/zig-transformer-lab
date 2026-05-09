# 01 — Zig 0.16.0 Primer for This Library

This chapter covers the Zig concepts you need to understand the code in
`zig-transformer-lab`.  It is NOT a general Zig tutorial — it focuses on
the specific language features and patterns used in this library, with
worked examples drawn from the actual source code.

If you already know Zig, skim for the 0.16-specific gotchas (marked with
**GOTCHA**). If you are new to Zig, read sequentially.

---

## 1. Why Zig for a Tensor Library?

Zig gives us three things that C and Python don't:

1. **Compile-time generics** — Our `Shape` struct uses a fixed-size `[4]usize`
   array with a `u2` rank field. No heap allocation for shapes, ever. In C
   you'd need a separate allocation or a max-dim constant; in Python every
   shape is a heap-allocated tuple.

2. **Explicit allocators** — Every function that allocates memory takes an
   `allocator` parameter. There is no global heap. This means we can use an
   arena for a forward pass (free everything at once), a GPA for tests (detect
   leaks), or a CUDA allocator for device memory — all without changing the
   calling code.

3. **No hidden control flow** — No operator overloading, no constructors, no
   destructors called implicitly. When you see `tensor.deinit(alloc)`, that's
   the only place the memory is freed. When you see `try someOp(...)`, the
   error handling is right there.

---

## 2. Basic Types You'll See Everywhere

### 2.1 Slices vs Arrays

```zig
// Array: fixed size, lives on the stack or inline in a struct
var dims: [4]usize = .{ 2, 3, 1, 1 };

// Slice: a pointer + length, can point into any memory
const data: []f32 = try allocator.alloc(f32, 100);
```

Our `Tensor.data` is a `[]f32` — a slice pointing to the heap-allocated
buffer. Our `Shape.dims` is a `[4]usize` — a fixed array embedded in the
struct (no allocation).

**GOTCHA:** In Zig, `[]T` is a slice (fat pointer: ptr + len), while
`[N]T` is an array (value type, N known at compile time). They are NOT
interchangeable. You can take a slice of an array with `dims[0..4]`, but
you cannot assign a slice to an array field.

### 2.2 Error Unions

Every fallible function in this library returns `!T` (error union):

```zig
pub fn zeros(allocator: Allocator, shape: Shape) !Tensor {
    const t = try Tensor.init(allocator, shape);
    return t;
}
```

The `!Tensor` return type means "either a Tensor or an error from the
current error set." Our library uses `LabError` (defined in
`src/core/errors.zig`) which contains:

- `OutOfMemory` — allocation failed
- `ShapeMismatch` — incompatible shapes
- `InvalidArgument` — out-of-range value
- `NotImplemented` — code path exists but not yet written
- `CudaError` — CUDA driver call failed (Stage 7)
- `IoError` — file I/O failure
- `NumericalError` — NaN/Inf detected

**GOTCHA:** `try expr` unwraps the error union or returns the error to the
caller. `catch` handles it inline. Never ignore errors — Zig has no
exceptions, so every error is explicit.

### 2.3 Optionals

```zig
pub const Tensor = struct {
    grad: ?*Tensor,        // null until backward() is called
    tape_node: ?NodeId,    // null if not part of a computation graph
};
```

The `?` prefix means "maybe null." You unwrap with `orelse` for a default
or `if (val) |v|` for conditional access:

```zig
// Provide a default
const g = tensor.grad orelse &zero_tensor;

// Conditional access
if (tensor.tape_node) |node_id| {
    // node_id is a NodeId, guaranteed non-null
    tape.backward(node_id);
}
```

**GOTCHA:** `?*Tensor` is "optional pointer to Tensor," NOT "pointer to
optional Tensor." The `?` binds to the entire type `*Tensor`. So `null`
means "no pointer," not "pointer to null."

---

## 3. Allocators

### 3.1 The Allocator Interface

Zig's `std.mem.Allocator` is an interface (a struct with a vtable pointer).
Every allocation function in this library takes it as the first parameter:

```zig
var t = try Tensor.init(allocator, Shape.init2D(2, 3));
defer t.deinit(allocator);
```

**Why pass the allocator to both init AND deinit?** Because the Tensor
struct doesn't store the allocator — it would add 16 bytes per tensor.
In a training loop that creates thousands of intermediate tensors, that
adds up. The caller already has the allocator; passing it twice is cheaper
than storing it.

### 3.2 Arena Allocator

For short-lived batches of allocations, the arena is ideal:

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    // ... create hundreds of tensors ...
    // No need to deinit — the arena frees everything at return
}
```

**GOTCHA:** `init.arena.allocator()` is the Zig 0.16.0 way to get an arena.
Do NOT use `std.heap.ArenaAllocator.init()` in examples — the `Init`
struct provides one automatically.

### 3.3 GPA (General Purpose Allocator)

For tests and long-running code, use GPA which detects leaks:

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
defer std.debug.assert(gpa.deinit() == .ok);
const allocator = gpa.allocator();
```

Our tests use `std.testing.allocator`, which wraps GPA and automatically
fails the test if any memory is leaked.

**GOTCHA:** In Zig 0.16.0, `var gpa = std.heap.GeneralPurposeAllocator(.{}).init;`
does NOT take an allocator parameter. The old `.init(allocator)` syntax
is stale.

---

## 4. Structs and Methods

### 4.1 Struct Definition

```zig
pub const Shape = struct {
    dims: [4]usize,
    rank: u2,

    pub fn ndim(self: Shape) usize {
        return @as(usize, self.rank) + 1;
    }

    pub fn init2D(d0: usize, d1: usize) Shape {
        return .{ .dims = .{ d0, d1, 1, 1 }, .rank = 1 };
    }
};
```

Key points:
- Fields are public by default (unlike C++ classes).
- `self` is not a keyword — it's just a parameter name convention.
- Methods that don't mutate take `self: Shape` (by value); mutating methods
  take `self: *Shape` (by pointer).

### 4.2 Constructor Pattern

Zig has no constructors. Factory functions return initialized structs:

```zig
// Rank-specific constructors (this is OUR pattern, not a language feature)
const s1 = Shape.init1D(10);
const s2 = Shape.init2D(2, 3);
const s3 = Shape.init3D(2, 3, 4);
const s4 = Shape.init4D(2, 3, 4, 5);
```

**GOTCHA:** There is NO `Shape.init(&.{2, 3})` constructor in this library.
If you try to use it, compilation will fail. See AGENTS.md for the full
rationale — this tripped up three separate AI agents.

### 4.3 Free Functions vs Methods

Some operations are free functions (not methods on the struct):

```zig
// Free function — takes both shapes as arguments
const same = shape.equals(a, b);

// NOT a method: a.equals(b) does not exist
```

Why? Because `equals` is symmetric — it doesn't "belong" to one shape
more than the other. Similarly, `broadcastShapes(a, b)` and
`computeStrides(shape)` are free functions.

---

## 5. comptime

### 5.1 What comptime Does

`comptime` means "evaluate at compile time." Zig's comptime is more powerful
than C++ constexpr — you can run arbitrary Zig code at compile time:

```zig
// This array length is computed at compile time
const label = switch (self) {
    .f32 => "f32",
};
```

In our library, comptime appears primarily in the `u2` rank field and
`@intCast` operations where we know the range is small.

### 5.2 When You'll See comptime

You won't write much comptime code in this library — the tensor operations
are runtime loops over `[]f32`. But you'll encounter it in:

- `@intCast(u2, value)` — casting to our rank type
- `switch` on enum variants (exhaustive — compiler checks all cases)
- `@compileError` for invalid configurations

---

## 6. Error Handling in Detail

### 6.1 try vs catch

```zig
// try: unwrap or propagate the error upward
var t = try Tensor.init(allocator, shape);

// catch: handle the error locally
var t = Tensor.init(allocator, shape) catch |err| switch (err) {
    error.OutOfMemory => return error.OutOfMemory,
    else => unreachable,
};
```

### 6.2 errdefer

`errdefer` runs only if the enclosing block exits with an error:

```zig
var out = try Tensor.init(allocator, out_shape);
errdefer out.deinit(allocator);
// If any line below returns an error, out.deinit(allocator) runs automatically.
// If we reach the return, out is returned and errdefer does NOT run.
```

This pattern is used in `softmax()`, `matmul()`, and `crossEntropy()` to
prevent leaks when intermediate operations fail.

### 6.3 Our Error Set

All errors are in `src/core/errors.zig`:

```zig
pub const LabError = error{
    ShapeMismatch,    // shapes are incompatible
    OutOfMemory,      // allocation failed
    InvalidArgument,  // out-of-range value
    CudaError,        // CUDA call failed
    IoError,          // file I/O failed
    NotImplemented,   // code path not yet written
    NumericalError,   // NaN/Inf detected
};
```

Every public function returns `!T` using this set. You can `catch` specific
errors or propagate them all with `try`.

---

## 7. defer and Ownership

### 7.1 defer: Guaranteed Cleanup

`defer` runs when the enclosing scope exits, regardless of how:

```zig
var t = try Tensor.init(allocator, Shape.init2D(2, 3));
defer t.deinit(allocator);
// t is valid here, and will be freed when the scope exits
```

**Execution order:** defers run in LIFO (last-in, first-out) order — like
a stack. If you have two defers, the second one runs first.

### 7.2 Ownership Rules in This Library

1. **Every owned tensor must be deinited.** `owned=true` means the tensor
   allocated its data buffer and must free it.

2. **View tensors (owned=false) must NOT be deinited.** They share another
   tensor's data. Only the original owner frees.

3. **The original tensor must outlive its views.** If you deinit the owner
   and then access a view, that's undefined behavior (dangling pointer).

```zig
var t = try Tensor.init(allocator, Shape.init2D(2, 3));
defer t.deinit(allocator);

const v = t.view();       // v.owned == false
const tr = try t.transpose2d(); // tr.owned == false
// DO NOT call v.deinit() or tr.deinit()
// t.deinit() frees the buffer that v and tr share
```

---

## 8. The `var` vs `const` Rule

**GOTCHA:** In Zig 0.16.0, if a local variable is never mutated after
initialization, you MUST use `const`. The compiler emits a hard error
(not a warning) if you use `var` unnecessarily:

```zig
// WRONG — compiler error: variable 'shape' is never mutated
var shape = Shape.init2D(2, 3);

// CORRECT — const because we never change it
const shape = Shape.init2D(2, 3);

// CORRECT — var because we mutate it
var sum: f32 = 0.0;
for (data) |v| sum += v;
```

This forces you to think about mutability. When you see `var`, you know
the value changes. When you see `const`, you know it's fixed.

**Rule of thumb:** Use `const` by default. Only change to `var` when the
compiler tells you to (or when you know you'll mutate).

---

## 9. Build System (build.zig)

### 9.1 Module System

Zig 0.16.0 uses `b.createModule()` and `root_module.addImport()`:

```zig
const lib_mod = b.addModule("zig_transformer_lab", .{
    .root_source_file = b.path("src/root.zig"),
    .target = target,
    .optimize = optimize,
});
```

Consumer files import with `@import("zig_transformer_lab")`.

**GOTCHA:** The old `b.addExecutable` + `exe.addModule` pattern is stale.
In 0.16.0, you create a module and pass it via `.imports` in the
executable's `root_module`.

### 9.2 Running Examples

```bash
zig build run-example -Dexample=01_tensor_playground
```

The build.zig resolves this to `examples/01_tensor_playground.zig`, creates
an executable, and runs it.

### 9.3 Build Options

```bash
zig build test -Dcuda=true         # Enable CUDA tests
zig build run-example -Dexample=01_tensor_playground
zig build docs                      # Print doc line counts
zig build kernels -Dcuda=true       # Compile .cu to .ptx
```

---

## 10. The Writer Interface

### 10.1 Printing to stdout

```zig
const writer = std.Io.get().writer();
try writer.print("shape = {s}\n", .{shapeStr});
```

**GOTCHA:** `std.Io.get().writer()` is the 0.16.0 way to get stdout.
The old `std.io.getStdOut().writer()` still works but `std.Io` is the
modern API.

### 10.2 Fixed-Buffer Writing

For tests and formatting into a local buffer:

```zig
var buf: [512]u8 = undefined;
var writer = std.Io.Writer.fixed(&buf);
try debugSummary(tensor, &writer);
const output = writer.buffered(); // returns []const u8 of what was written
```

### 10.3 Format Specifiers

```zig
try writer.print("{d}", .{value});      // decimal integer
try writer.print("{d:.3}", .{float_val}); // float with 3 decimal places
try writer.print("{s}", .{str_slice});   // string slice
try writer.print("{}", .{bool_val});     // boolean (true/false)
```

---

## 11. Integer Casting

### 11.1 The New Builtins

Zig 0.16.0 renamed the cast builtins. The old names are gone:

| Old (pre-0.16) | New (0.16.0) | Purpose |
|----------------|--------------|---------|
| `@intToPtr` | `@ptrFromInt` | integer → pointer |
| `@ptrToInt` | `@intFromPtr` | pointer → integer |
| `@enumToInt` | `@intFromEnum` | enum → integer |
| `@intToEnum` | `enumFromInt` | integer → enum |

**GOTCHA:** The compiler will NOT suggest the new names if you use the old
ones. It just says "error: use of undeclared identifier." Memorize the
table above or keep it handy.

### 11.2 @intCast, @floatCast, @intFromFloat

Narrowing casts require explicit builtins:

```zig
// usize → u2 (narrowing, must use @intCast)
const rank: u2 = @intCast(ndim - 1);

// f64 → f32 (narrowing, must use @floatCast)
const val: f32 = @floatCast(f64_val);

// f32 → usize (converting a class index stored as f32)
const idx: usize = @intFromFloat(@round(targets.data[i]));
```

**GOTCHA:** `@intCast` can panic at runtime if the value doesn't fit. In
our library, we guard with bounds checks before casting (e.g., checking
`class_idx < C` before using it as an index).

---

## 12. Calling Convention

### 12.1 C Interop

When calling C functions (CUDA bindings in Stage 7):

```zig
pub const cudaInit_t = *const fn () callconv(.c) c_int;
```

**GOTCHA:** `callconv(.c)` not `callconv(.C)`. The lowercase `.c` is the
0.16.0 syntax. The uppercase `.C` was valid in older Zig but is now
rejected by the compiler.

### 12.2 Main Function Signature

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    // ...
}
```

**GOTCHA:** The old `pub fn main() !void` signature is stale. 0.16.0
passes `std.process.Init` which contains the arena and other startup data.

---

## 13. ArrayList

### 13.1 Unmanaged API

Zig 0.16.0 uses the "unmanaged" ArrayList style where you pass the
allocator to each method:

```zig
var list: std.ArrayList(u32) = .empty;
try list.append(gpa, 42);
try list.append(gpa, 99);
defer list.deinit(gpa);
```

**GOTCHA:** The old `ArrayList(T).init(allocator)` + `list.append(item)`
is stale. In 0.16.0:
- Initialize with `.empty` (not `.init(allocator)`)
- Pass allocator to each method: `list.append(gpa, item)`
- Deinit with `list.deinit(gpa)`

The Stage 3 autograd tape will use `ArrayList(Node)` heavily.

---

## 14. Testing

### 14.1 Co-located Tests

Tests live in the same file as the code they test:

```zig
// src/tensor/shape.zig

pub fn totalElements(shape: Shape) usize {
    var product: usize = 1;
    for (shape.dims) |d| product *= d;
    return product;
}

test "totalElements" {
    try std.testing.expectEqual(@as(usize, 6), totalElements(Shape.init2D(2, 3)));
}
```

### 14.2 Test Discovery

Zig's test runner only discovers `test "..."` blocks in files that are
transitively imported by the test root module. Our `src/root.zig` has:

```zig
test {
    _ = shape;
    _ = tensor_mod;
    _ = ops.create;
    // ... each sub-module
}
```

**GOTCHA:** `tests/unit_all.zig` is dead code — it is NOT wired into
`build.zig` and would fail to compile if used. Don't add tests there.
Add them co-located in the source file.

### 14.3 Approximate Equality

For floating-point comparisons:

```zig
try std.testing.expectApproxEqAbs(@as(f32, 0.0900), out.data[0], 1e-3);
```

The third argument is the tolerance. Use `1e-3` for most tensor ops
(softmax, matmul) and `1e-4` for simple elementwise ops.

### 14.4 Memory Leak Detection

`std.testing.allocator` wraps GPA and detects leaks. If any test leaks
memory, the entire test run fails:

```zig
var t = try Tensor.init(std.testing.allocator, shape);
defer t.deinit(std.testing.allocator);
// If you forget the defer, the test fails with "memory leak detected"
```

---

## 15. @memset and @memcpy

### 15.1 Bulk Memory Operations

Zig 0.16.0 provides these as builtins (not functions):

```zig
// Fill every element with a value
@memset(tensor.data, 0);          // all zeros
@memset(tensor.data, @as(f32, 1.0)); // all ones

// Copy all elements
@memcpy(dst.data, src.data);      // dst and src must be same length
```

**Why use @memset instead of a loop?** The compiler can vectorize @memset
into SIMD stores (e.g., AVX2 stores 8 f32 values at once). A manual loop
would need explicit SIMD or rely on auto-vectorization.

**GOTCHA:** `@memcpy` requires that dst and src are non-overlapping slices.
For overlapping copies, use `std.mem.copyForwards` or `copyBackwards`.

---

## 16. Switch and Exhaustiveness

### 16.1 Exhaustive Matching

Zig's `switch` must cover every case. This is especially useful with enums:

```zig
pub fn label(self: DType) []const u8 {
    return switch (self) {
        .f32 => "f32",
        // If we add .f16, the compiler will error here until we add a case
    };
}
```

This means: if you add a new `DType` variant (e.g., `.f16`), every
`switch` on `DType` in the entire codebase will fail to compile until
you handle the new variant. No silent fallthrough.

### 16.2 Switch on Error Unions

```zig
const data = allocator.alloc(f32, n) catch |err| switch (err) {
    error.OutOfMemory => return error.OutOfMemory,
};
```

---

## 17. The `@abs`, `@max`, `@min` Builtins

Zig 0.16.0 uses builtins for min/max/abs instead of functions:

```zig
const max_val = @max(a, b);     // works for integers, floats
const abs_val = @abs(x);        // works for integers, floats
```

In our softmax, we find the maximum per row with:

```zig
max_val = @max(max_val, val);
```

**GOTCHA:** `std.math.max` and `std.math.min` still exist for comparing
more than two values, but for the common two-argument case, use `@max`
and `@min`.

---

## 18. Working with f32

### 18.1 IEEE 754 Semantics

Our library uses f32 exclusively (decision D9). Key behaviors:

- `1.0 / 0.0` = +Inf (not an exception)
- `0.0 / 0.0` = NaN
- `-0.0` exists and is distinct from `+0.0`
- NaN comparisons are ALWAYS false: `NaN == NaN` is false, `NaN < 1.0` is false

### 18.2 Checking for NaN and Inf

```zig
if (std.math.isNan(v)) { /* handle NaN */ }
if (std.math.isInf(v)) { /* handle Inf */ }
if (std.math.isFinite(v)) { /* neither NaN nor Inf */ }
```

### 18.3 Special Values

```zig
const neg_inf = -std.math.inf(f32);    // -∞
const pos_inf = std.math.inf(f32);     // +∞
const nan_val = std.math.nan(f32);     // NaN
```

Our softmax initializes `max_val` to `-inf(f32)` so that any real value
is larger:

```zig
var max_val: f32 = -std.math.inf(f32);
```

---

## 19. File Organization

### 19.1 Module Structure

```
src/
├── root.zig          # re-exports everything
├── core/              # foundational types
│   ├── errors.zig
│   ├── dtype.zig
│   ├── device.zig
│   └── rng.zig
└── tensor/           # tensor struct and ops
    ├── shape.zig
    ├── tensor.zig
    ├── print.zig
    └── ops/
        ├── create.zig
        ├── elementwise.zig
        ├── reduce.zig
        ├── matmul.zig
        ├── unary.zig
        ├── softmax.zig
        └── loss.zig
```

### 19.2 Import Convention

Files import using named modules registered in `build.zig`:

```zig
const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Shape = @import("../shape.zig").Shape;
```

Consumer code (examples, tests) imports the library as one module:

```zig
const ztl = @import("zig_transformer_lab");
const Shape = ztl.shape.Shape;
```

---

## 20. Zig 0.16.0 Gotchas Summary

| # | Gotcha | Fix |
|---|--------|-----|
| 1 | `std.fs.cwd()` doesn't exist | Use `std.Io.Dir.cwd()` |
| 2 | `var` on non-mutated locals is an error | Use `const` by default |
| 3 | Unused function parameters are errors | Prefix with `_` or remove |
| 4 | `build.zig.zon` fingerprint must match | Let the compiler tell you |
| 5 | Unused `b.option()` values are errors | Add `_ = varname;` |
| 6 | `addSystemCommand` arg types | Must be `[]const []const u8` |
| 7 | `LazyPath` syntax | Use `.cwd_relative = "path"` |
| 8 | `exe.builder.allocator` doesn't exist | Use `b.allocator` |
| 9 | Test discovery requires module imports | Add `_ = module;` in root.zig |
| 10 | `std.io.fixedBufferStream` may not exist | Use `std.fmt.bufPrint` or `Writer.fixed` |
| 11 | `callconv(.C)` is stale | Use `callconv(.c)` |
| 12 | `@intToPtr` etc. are stale | Use `@ptrFromInt`, `@intFromPtr`, `@intFromEnum` |
| 13 | `main() !void` is stale | Use `main(init: std.process.Init) !void` |
| 14 | `ArrayList(T).init(a)` is stale | Use `var list: ArrayList(T) = .empty;` |
| 15 | `addExecutable` + `addModule` is stale | Use `b.createModule` + `addImport` |

---

## 21. Common Mistakes

1. **Using `Shape.init(&.{2, 3})`** — This constructor does not exist.
   Use `Shape.init2D(2, 3)`. This is the #1 mistake; see AGENTS.md.

2. **Forgetting `try` on fallible functions** — If a function returns
   `!T`, you must use `try` or `catch`. Silent error discarding is not
   allowed.

3. **Calling `deinit` on a view tensor** — View tensors (owned=false) share
   another tensor's buffer. Only the owning tensor should call deinit.

4. **Using `var` for everything** — Zig 0.16.0 requires `const` for
   non-mutated locals. The compiler error is clear: change to `const`.

5. **Comparing `rank` to a dimension count** — `rank` stores `ndim - 1`.
   Use `shape.ndim()` instead of comparing `rank` directly.

6. **Stale API patterns** — The gotcha table above covers the most common
   ones. When in doubt, check `skills/modern-zig-0-16-tutor/SKILL.md`.

7. **Forgetting `errdefer`** — If you allocate a tensor and then call a
   fallible function, use `errdefer t.deinit(alloc)` to prevent leaks
   when the fallible function errors.

8. **Mixing up `@intCast` and `@intFromFloat`** — `@intCast` is for
   integer-to-integer narrowing. `@intFromFloat` is for float-to-integer
   conversion. They are NOT interchangeable.

9. **Using `std.testing.allocator` in examples** — Use `init.arena.allocator()`
   in `main`. `std.testing.allocator` is only for `test` blocks.

10. **Writing `tests/unit_all.zig`** — That file is dead code. Add tests
    co-located in the source file, and they'll be discovered automatically
    through `src/root.zig`.
