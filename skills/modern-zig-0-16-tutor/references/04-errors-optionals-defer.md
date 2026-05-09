# 04 — Errors, optionals, `defer` / `errdefer` (0.16.0)

Zig's error-handling discipline is what makes correct-by-default code possible.
Internalize this before you write anything nontrivial.

## Error sets and error unions

- An **error set** is an enum-like type: `const NetErr = error{ Closed, Reset };`.
- An **error union** is `ErrSet!Payload` (or `!Payload` for inferred sets).
- `error.Foo` is sugar for a single-variant set.
- `||` merges two error sets; duplicate names collapse to the same id.

```zig
const ParseErr = error{ InvalidChar, Overflow };

fn parseU8(s: []const u8) ParseErr!u8 {
    var x: u16 = 0;
    for (s) |c| {
        if (c < '0' or c > '9') return error.InvalidChar;
        x = x * 10 + (c - '0');
        if (x > 255) return error.Overflow;
    }
    return @intCast(x);
}
```

## Handling error unions

```zig
// `try` = unwrap or propagate
const n = try parseU8("42");

// `catch` = provide a default
const n2 = parseU8("abc") catch 0;

// `catch |err| ...` = capture and decide
const n3 = parseU8("abc") catch |err| switch (err) {
    error.InvalidChar => 0,
    error.Overflow => 255,
};

// `if (expr) |v| ... else |err| ...`
if (parseU8("12")) |v| {
    std.debug.print("got {d}\n", .{v});
} else |err| {
    std.debug.print("err = {s}\n", .{@errorName(err)});
}
```

## `try` = `catch |e| return e`

`try expr` is exactly `expr catch |e| return e;` with the inferred error set
widened to include `e`.

## Inferred error sets with `!T`

```zig
fn f() !i32 {   // inferred error set
    return try g();
}
```

Rules:
- OK for internal / private functions.
- Avoid on **public** APIs with semver surface — add a named set instead.
- Avoid on **recursive** functions and function-pointer targets — inferred
  sets can diverge.

## `defer` and `errdefer`

- `defer expr;` runs at end of scope in reverse declaration order.
- `errdefer expr;` runs only if the scope exits via an **error**.
- `errdefer |err| { ... }` captures the error value for logging / teardown.

```zig
fn loadConfig(a: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(path, .{});
    defer f.close();

    const buf = try a.alloc(u8, 4096);
    errdefer a.free(buf);   // only on error path

    const n = try f.readAll(buf);
    if (n == 0) return error.EmptyConfig;

    return buf[0..n];
}
```

**Mentor note.** A missing `errdefer` on an allocated buffer is one of the
top 3 Zig review findings. `grep_stale_patterns.py` cannot catch this —
the reviewer must.

## Optionals

- `?T` means "T or null"; null-optimized for pointer types (same size as `*T`).
- Unwrap with `if (maybe) |v| { ... } else { ... }` or `maybe orelse default`.
- `.?` is shorthand for `orelse unreachable` — asserts non-null in Debug.

```zig
const home: ?[]const u8 = init.environ_map.get("HOME");
const h = home orelse return error.HomeMissing;
```

## `@typeInfo` is snake_case now

```zig
// WRONG
@typeInfo(@TypeOf(foo)).ErrorUnion.payload

// CORRECT
@typeInfo(@TypeOf(foo)).error_union.payload
// Other examples:
// @typeInfo(T).@"struct".fields
// @typeInfo(T).pointer.child
// @typeInfo(T).@"fn".return_type
```

## Error return traces

Enabled by default in Debug. On an error return, Zig records every call site
the error bubbled through — richer than a stack trace, which only has the
innermost frame.

```zig
fn top() !void {
    return inner();
}
fn inner() !void {
    return error.Sad;
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    top() catch |err| {
        std.debug.dumpStackTrace(@errorReturnTrace().?);
        return err;
    };
}
```

## Common review findings

- **Missing `try`.** Silent error drop on an expression statement.
- **`catch {}`.** Empty catch swallows errors — flag always.
- **Wrong `errdefer` ordering.** `errdefer` must come *after* the allocation
  it cleans up, and *before* any further fallible call.
- **`anyerror`.** Avoid; prefer explicit or inferred sets.
- **`orelse unreachable` (via `.?`) in library code.** Turn into an explicit
  error if the null represents a real failure mode.

## Common mentor diagnostic questions

- "What error values does this function return? Is the set meaningful to the
  caller?"
- "On the error path, do we free everything allocated so far?"
- "Should this be `try` or should we handle the error locally?"

<!-- ~2.0k tokens · verified against Zig 0.16.0 language reference -->
