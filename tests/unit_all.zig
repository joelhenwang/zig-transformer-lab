//!
//! tests/unit_all.zig — Test aggregator
//!
//! Purpose:
//!   Zig's test runner only discovers `test` blocks in files that are
//!   transitively @imported by the test root. This file imports the library
//!   module, which re-exports every source file under src/, so that
//!   `zig build test` runs all unit tests without manual registration.
//!
//! How to maintain:
//!   When a new .zig file is added under src/, add a re-export in
//!   src/root.zig. The tests in that file will be automatically discovered
//!   through the module import chain.
//!
//! Memory:
//!   All tests use `std.testing.allocator` which detects leaks.
//!

const std = @import("std");
const ztl = @import("zig_transformer_lab");

test "unit_all smoke test" {
    try std.testing.expect(true);
}
