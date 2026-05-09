//!
//! tests/unit_all.zig — Test aggregator
//!
//! Purpose:
//!   Zig's test runner only discovers `test` blocks in files that are
//!   transitively @imported by the test root. This file imports every
//!   source module directly so that `zig build test` runs all unit tests
//!   without manual registration.
//!
//! How to maintain:
//!   When a new .zig file is added under src/, add a @import here AND
//!   add the module to build.zig's test imports list so the import resolves.
//!
//! Memory:
//!   All tests use `std.testing.allocator` which detects leaks.
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");

// Direct imports of sub-modules so their co-located test blocks are
// discovered by the Zig test runner. The `test { _ = ...; }` declarative
// block forces the compiler to analyze each module, finding its tests.
// These module names must match the imports registered in build.zig.
const mod_errors = @import("errors");
const mod_dtype = @import("dtype");
const mod_device = @import("device");
const mod_rng = @import("rng");
const mod_shape = @import("shape");
const mod_tensor = @import("tensor");
const mod_print = @import("print");

test {
    _ = mod_errors;
    _ = mod_dtype;
    _ = mod_device;
    _ = mod_rng;
    _ = mod_shape;
    _ = mod_tensor;
    _ = mod_print;
}

test "unit_all smoke test" {
    try std.testing.expect(true);
}
