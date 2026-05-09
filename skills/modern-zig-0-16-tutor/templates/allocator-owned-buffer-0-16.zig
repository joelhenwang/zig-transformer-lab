//! target: Zig 0.16.0 — illustrative, verify with `zig test` on 0.16.0.
//!
//! `Buffer` — owned slice of bytes, init/deinit with an allocator, errdefer
//! discipline. Templates a common allocator-ownership pattern for 0.16.
//!
//! Idioms demonstrated:
//!   - `std.mem.Allocator` parameter on init
//!   - `errdefer a.free(buf)` placed immediately after the allocation
//!   - `deinit` takes the same allocator that was passed to `init`
//!   - `self.* = undefined;` in `deinit` to poison reuse in Debug
//!   - public API takes `allocator` explicitly — never stored

const std = @import("std");

pub const Buffer = struct {
    data: []u8,

    /// Create a zero-initialized buffer of `n` bytes. Transfers ownership
    /// of the memory to the caller. Caller must call `deinit(a)` with the
    /// same `a` used here.
    pub fn init(a: std.mem.Allocator, n: usize) !Buffer {
        const data = try a.alloc(u8, n);
        errdefer a.free(data);

        @memset(data, 0);

        return .{ .data = data };
    }

    /// Release memory. The buffer must not be used afterwards.
    pub fn deinit(self: *Buffer, a: std.mem.Allocator) void {
        a.free(self.data);
        self.* = undefined;
    }

    pub fn fill(self: *Buffer, byte: u8) void {
        @memset(self.data, byte);
    }

    pub fn len(self: Buffer) usize {
        return self.data.len;
    }
};

test "Buffer init/deinit with testing allocator" {
    const a = std.testing.allocator;
    var b = try Buffer.init(a, 16);
    defer b.deinit(a);
    try std.testing.expectEqual(@as(usize, 16), b.len());
    for (b.data) |x| try std.testing.expectEqual(@as(u8, 0), x);
}

test "Buffer.fill overwrites all bytes" {
    const a = std.testing.allocator;
    var b = try Buffer.init(a, 8);
    defer b.deinit(a);
    b.fill(0xAB);
    for (b.data) |x| try std.testing.expectEqual(@as(u8, 0xAB), x);
}

// TODO(learner): add a `copyFrom(src: []const u8) !void` method that
// returns `error.OutOfRange` if src doesn't fit. Write a test for the
// error path. Use expectError.

// TODO(learner): add a `resize(a: Allocator, new_len: usize) !void` method
// using `a.realloc`. Consider the edge cases: shrinking (cheap), growing
// with relocation (may invalidate pointers into `data`).
