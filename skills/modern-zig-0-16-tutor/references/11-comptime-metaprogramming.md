# 11 — comptime and metaprogramming (0.16.0)

`comptime` lets Zig express generics and reflection without a separate
template language. The critical 0.16 delta: **`@Type` is gone**, replaced by
eight individual type-building builtins. And `@typeInfo` field names are
**snake_case**.

## `comptime` in four shapes

| Where | Meaning |
|---|---|
| `fn f(comptime T: type, x: T) T` | Parameter must be known at compile time |
| `comptime var i = 0;` | Variable exists only at compile time |
| `comptime { ... }` | Block executed entirely at compile time |
| `comptime expr` | Force this expression to be a compile-time value |

Inside a comptime context:
- All values are comptime-known.
- `if` / `while` / `for` / `switch` are unrolled.
- No `return` / `try` unless the enclosing function is being comptime-called.
- No runtime side-effects (no extern calls, no volatile I/O).
- Branch quota enforced (`@setEvalBranchQuota(new_n)`).

## Generic data structure pattern

```zig
fn LinkedList(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            value: T,
            next: ?*Node,
        };

        head: ?*Node = null,

        pub fn prepend(self: *Self, a: std.mem.Allocator, v: T) !void {
            const n = try a.create(Node);
            n.* = .{ .value = v, .next = self.head };
            self.head = n;
        }
    };
}
```

## `@Type` → eight replacement builtins (0.16)

```zig
const U10      = @Int(.unsigned, 10);         // u10
const I128     = @Int(.signed, 128);          // i128
const Tup      = @Tuple(&.{ u32, [2]f64 });   // tuple type
const PtrTo32  = @Pointer(.one, .{}, u32, null);
const FnType   = @Fn(&.{ i32, i32 }, &.{ .{}, .{} }, i32, .{});

const TagInt   = u8;
const Color    = @Enum(TagInt, .exhaustive, &.{ "red", "green", "blue" }, &.{ 0, 1, 2 });

const Record   = @Struct(.auto, null,
    &.{ "id", "score" },
    &.{ u32, f32 },
    &@splat(.{}),  // default attributes for both fields
);

const MyUnion  = @Union(.auto, u8,
    &.{ "none", "some" },
    &.{ void, u32 },
    &@splat(.{}),
);

// Reserved literal — rare but needed for generic code
const EnumLit = @EnumLiteral();
```

What's **not** replaced (use normal syntax instead):
- `@Float(N)` → use `std.meta.Float(N)`.
- `@Array(len, T)` → write `[len]T` or `[len:s]T`.
- `@Opaque()` → write `opaque {}`.
- `@Optional(T)` → write `?T`.
- `@ErrorUnion(E, T)` → write `E!T`.
- `@ErrorSet(&.{"A","B"})` → **removed entirely**; declare `error{ A, B }`.

## `@typeInfo` is snake_case

```zig
// WRONG
@typeInfo(T).Struct.fields
@typeInfo(T).Pointer.child
@typeInfo(T).ErrorUnion.payload

// CORRECT
@typeInfo(T).@"struct".fields
@typeInfo(T).pointer.child
@typeInfo(T).error_union.payload
```

Other renames to expect: `.@"enum"`, `.@"fn"`, `.@"union"`, `.array`,
`.optional`, `.int`, `.float`.

## `inline for` / `inline while`

Loops with comptime-known bounds can be forced to unroll:

```zig
fn sumTuple(t: anytype) i64 {
    var sum: i64 = 0;
    inline for (@typeInfo(@TypeOf(t)).@"struct".fields) |f| {
        sum += @field(t, f.name);
    }
    return sum;
}
```

## `@setEvalBranchQuota`

Raise the branch budget for a heavy comptime computation:

```zig
comptime {
    @setEvalBranchQuota(10_000);
    // ... long constant folding
}
```

Default quota is 1000. Errors of the form
`"evaluation exceeded 1000 backwards branches"` mean you need this.

## Comptime-only vs runtime types

A type is **comptime-only** if one of its members is itself comptime-only
(`type`, function types, etc.). 0.16 relaxed the rule: **pointers to
comptime-only types are not themselves comptime-only** — e.g.
`[]const std.builtin.Type.StructField` can flow at runtime; you just can't
load the aggregate at runtime.

## Common pitfalls and the modern fix

| Pitfall | Fix |
|---|---|
| `@Type(.{ .int = ... })` | `@Int(.unsigned, N)` |
| `@Type(.{ .@"struct" = ... })` | `@Struct(...)` (or just write the struct literal) |
| `@typeInfo(T).Struct` | `@typeInfo(T).@"struct"` |
| `usingnamespace mixin;` | Explicit re-export: `pub const foo = mixin.foo;` |
| Returning `&local_var` from a generic | Restructure to return by value, or accept `*out: T` |
| Exceeding branch quota silently | `@setEvalBranchQuota(N)` with an honest N |

## `usingnamespace` is gone

Replace with explicit re-exports:

```zig
// WRONG
pub usingnamespace @import("impl.zig");

// CORRECT
const impl = @import("impl.zig");
pub const foo = impl.foo;
pub const Bar = impl.Bar;
```

## Common mentor diagnostic questions

- "Does this really need to be comptime? Would runtime generation work?"
- "What's the compile-time cost? (branch quota / comptime memory)"
- "Is this reflection reading a comptime-only field or a runtime-usable one?"
- "Can you express this with a normal `comptime T: type` parameter instead
  of building a type from scratch with `@Struct`?"

<!-- ~2.4k tokens · verified against Zig 0.16.0 release notes -->
