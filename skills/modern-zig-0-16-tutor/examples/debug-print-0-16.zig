//! target: Zig 0.16.0 — illustrative; run with:
//!   zig build-exe debug-print-0-16.zig && .\debug-print-0-16.exe
//!
//! `std.debug.print` writes to stderr and does not require `io`. Use it
//! for ad-hoc debugging. No `init: std.process.Init` needed.

const std = @import("std");

pub fn main() void {
    const n: u32 = 7;
    const name: []const u8 = "world";
    std.debug.print("debug: hello {s}, n = {d}\n", .{ name, n });
}
