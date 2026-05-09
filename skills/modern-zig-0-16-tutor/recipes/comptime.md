# Recipes: comptime

## Build a tiny generic `Pair(A, B)` with `@Struct` (0.16 builtin)

**Problem.** Build a tuple-like generic `Pair(A, B)` type without `@Type`.

**Mentor explanation.** In 0.16, `@Type` is replaced by 8 builtins. For a
custom struct, `@Struct(layout, backingInt, fieldNames, fieldTypes,
fieldAttrs)` is the direct replacement. But **often the simplest path is to
return a struct literal from a generic function** — that's idiomatic Zig
and avoids reflection entirely.

**Minimal snippet — idiomatic approach.**
```zig
fn Pair(comptime A: type, comptime B: type) type {
    return struct {
        first: A,
        second: B,

        pub fn init(a: A, b: B) @This() { return .{ .first = a, .second = b }; }
    };
}
```

**Minimal snippet — reflection approach (when dynamic).**
```zig
fn makePair(comptime A: type, comptime B: type) type {
    return @Struct(
        .auto,
        null,
        &.{ "first", "second" },
        &.{ A, B },
        &@splat(.{}),
    );
}
```

**Test to write.**
```zig
test "Pair(i32, f64) works" {
    const P = Pair(i32, f64);
    const p = P.init(7, 3.14);
    try std.testing.expectEqual(@as(i32, 7), p.first);
    try std.testing.expectEqual(@as(f64, 3.14), p.second);
}
```

**Stale pattern to avoid.**
```zig
// WRONG
const P = @Type(.{ .@"struct" = .{
    .layout = .auto,
    .fields = &.{ .{ .name = "first", .type = i32, ... } },
    .decls = &.{},
    .is_tuple = false,
}});
```

**Exercise.** Write `Triple(A, B, C)` using the idiomatic approach. Can you
add a `fn swap(self: @This()) Pair(B, A)`? Why would you avoid it on a
generic struct?

## `@typeInfo` is snake_case now

```zig
test "inspect fields" {
    const P = Pair(u32, []const u8);
    inline for (@typeInfo(P).@"struct".fields) |f| {
        std.debug.print("field: {s} of type {}\n", .{ f.name, f.type });
    }
}
```

`.@"struct"` — the escaped identifier form is required because `struct`
is a keyword. Same for `.@"enum"`, `.@"fn"`, `.@"union"`.

## Branch quota

Heavy comptime code may exceed the default 1000-branch budget:

```zig
comptime {
    @setEvalBranchQuota(50_000);
    // ... big constant table
}
```

<!-- ~1.0k tokens -->
