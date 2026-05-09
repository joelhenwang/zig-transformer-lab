# 10 — Testing and debugging (0.16.0)

Zig's built-in test runner is first-class. Use it aggressively, especially
for numerical / ML code where regressions are silent.

## Declaring tests

```zig
const std = @import("std");

// Named test — implicitly anyerror!void; cannot change return type
test "addOne adds one to 41" {
    try std.testing.expect(addOne(41) == 42);
    try std.testing.expectEqual(42, addOne(41));
}

// Doctest form — bound to a declaration, shown in generated docs
test addOne {
    try std.testing.expectEqual(42, addOne(41));
}

fn addOne(n: i32) i32 { return n + 1; }
```

## Running tests

```pwsh
zig test src/main.zig             # single file
zig build test                    # via build.zig test step
zig build test --test-filter "ten" # only tests whose name contains "ten"
zig build test --test-timeout 500ms  # 0.16 feature: per-test wall-clock limit
```

## Leak-detecting allocator

```zig
test "no leak" {
    const a = std.testing.allocator;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    try list.append(a, 1);
    try list.append(a, 2);
    try std.testing.expectEqual(2, list.items.len);
}
```

If a test leaks, the runner prints the leaked address and stack trace. This
is among the most valuable features of Zig testing.

## Useful expectations

| Call | Purpose |
|---|---|
| `expect(bool)` | Generic true-check |
| `expectEqual(expected, actual)` | Coerces `actual` to `@TypeOf(expected)` |
| `expectEqualStrings(a, b)` | `[]const u8` pretty diff |
| `expectEqualSlices(T, a, b)` | Slice equality with element pretty-print |
| `expectError(error.X, expr)` | Asserts `expr` returns `error.X` |
| `expectApproxEqAbs(a, b, tol)` | Float equality within absolute tolerance |
| `expectApproxEqRel(a, b, tol)` | Float equality within relative tolerance |
| `return error.SkipZigTest;` | Skip this test |

## Discovering tests in other files

```zig
test {
    // Pull in tests from sibling files
    _ = @import("util.zig");
    _ = @import("tensor.zig");
}
```

The anonymous `test` block references those files, so the compiler discovers
and runs their tests too.

## `OOM` path testing

```zig
test "handles OOM" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
        .fail_index = 0,  // fail the first allocation
    });
    const a = failing.allocator();
    try std.testing.expectError(error.OutOfMemory, concat(a, "a", "b"));
}
```

## Numerical comparison (ML-relevant)

```zig
test "matmul 2x2 identity" {
    const a = std.testing.allocator;
    var A = try Tensor2D.init(a, 2, 2);  defer A.deinit(a);
    var I = try Tensor2D.init(a, 2, 2);  defer I.deinit(a);
    var out = try Tensor2D.init(a, 2, 2); defer out.deinit(a);

    I.atPtr(0, 0).* = 1; I.atPtr(1, 1).* = 1;
    A.atPtr(0, 0).* = 3; A.atPtr(0, 1).* = 5;
    A.atPtr(1, 0).* = 7; A.atPtr(1, 1).* = 11;

    try matmulNaive(A, I, &out);
    try expectSlicesClose(f32, A.data, out.data, 1e-6);
}

fn expectSlicesClose(comptime T: type, a: []const T, b: []const T, tol: T) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (a, b, 0..) |x, y, i| {
        if (@abs(x - y) > tol) {
            std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.NotClose;
        }
    }
}
```

## Debugging tactics

- **Stack traces on crash.** 0.16 improves segfault handling across most
  targets. Running with `-O Debug` gets the richest traces.
- **Error return traces.** Free in Debug; nothing to enable. Dump with
  `std.debug.dumpStackTrace(@errorReturnTrace().?);`.
- **`std.debug.print`.** Stderr, immediate, no `io` needed. Prefer this over
  toy `std.log` for quick bisects.
- **`zig fmt` before debugging.** Reformat can flush out visual bugs like
  mis-nested blocks.
- **`@compileLog(...)`.** Prints a value at compile time; very useful when
  you can't see what type a generic is deducing.
- **Sanitize.** `zig build -Dsanitize-thread=true` / `-fstack-check`. Useful
  on Linux for race hunting; less useful on Windows.

## Common review findings

- Test that uses `gpa` / `arena` but not `std.testing.allocator` — leaks
  pass silently. Flag.
- Test that mutates global state and does not reset it. Flag.
- Numeric test using `==` on `f32` / `f64`. Flag; use `expectApproxEqAbs`.
- Test returning `!void` but calling only `expectEqual(...)` without `try`.
  Flag.
- Test named "it works" — rename to describe the behavior checked.

## Common mentor diagnostic questions

- "What's the minimal failing input? Can you pin it in a test before fixing?"
- "Is this deterministic? Any time-dependent or randomness source?"
- "Tolerance — absolute or relative? Why this number?"
- "If this test passes, what could still be broken?"

<!-- ~2.2k tokens · verified against Zig 0.16.0 langref -->
