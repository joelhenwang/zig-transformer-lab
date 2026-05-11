//!
//! zig-transformer-lab — Deterministic random number generator
//!
//! Purpose:
//!   Wraps `std.Random.Xoshiro256` in a struct that forces every consumer to
//!   supply an explicit seed.  This satisfies locked decision D14: "every
//!   random operation takes an explicit seed; runs must be deterministic given
//!   the same seed."  There is no global RNG — each training loop, each weight
//!   initializer, and each test creates its own `Rng` with a known seed.
//!
//! Shape contract:
//!   Rng carries no shape information. It produces scalar f32 values that
//!   callers then scatter into tensors.
//!
//! Math:
//!   uniform [0, 1):
//!     Map the generator's raw u64 output to f32 in [0, 1) via the
//!     standard library's `float(f32)` method.
//!
//!   Box-Muller standard normal N(0, 1):
//!     Given two independent uniforms u1 ∈ (0, 1] and u2 ∈ [0, 1),
//!       z0 = sqrt(-2 * ln(u1)) * cos(2 * π * u2)
//!       z1 = sqrt(-2 * ln(u1)) * sin(2 * π * u2)
//!     Both z0 and z1 are independent standard normals.  We return z0 and
//!     cache z1 ("spare") so the next call is free — this halves the cost of
//!     generating long sequences of normals.
//!
//!     We use `1.0 - floatf32()` for u1 to guarantee u1 > 0 (avoiding ln(0)
//!     which would produce -inf).  The cost is a negligible bias at the
//!     extreme ends of the f32 mantissa — acceptable for training.
//!
//! Memory ownership:
//!   Rng is a value type. No heap allocation.  The caller owns the Rng
//!   struct (typically on the stack or as a field of a larger struct).
//!
//! Error conditions:
//!   None.  All methods are infallible — the seed is always valid (any u64).
//!
//! TODO:
//!   - future: a CUDA-side curand generator with matching seed so
//!     that GPU-side sampling (e.g. dropout, MC sampling) can be
//!     reproducible against a CPU reference.
//!   - future: a `normalF32Slice` bulk method for weight init that
//!     avoids per-element call overhead.
//!
//! Credits:
//!   Box-Muller transform is standard (Box & Muller, 1958).  The spare-value
//!   caching trick is from NumPy's legacy implementation (now uses Ziggurat).
//!   No code copied.

const std = @import("std");

/// A deterministic random number generator seeded from an explicit u64.
///
/// Usage:
///   var rng = Rng.init(1337);
///   const x = rng.floatf32();    // uniform f32 in [0, 1)
///   const z = rng.normalF32();   // standard normal f32
///
/// Worked example:
///   var rng = Rng.init(42);
///   const a = rng.floatf32();    // e.g. 0.373...
///   const b = rng.floatf32();   // e.g. 0.718...
///   // Re-seeding with 42 produces the exact same sequence.
///
/// Memory: no allocation.  Rng is a plain struct the caller owns.
pub const Rng = struct {
    /// The underlying Xoshiro256 generator.  We wrap it rather than use it
    /// directly so that (a) we control the API surface and (b) we can swap
    /// the generator later without touching call sites.
    xoshiro: std.Random.Xoshiro256,

    /// Cached second normal from the Box-Muller pair.  `null` means we need
    /// to generate a fresh pair on the next `normalF32()` call; non-null
    /// means we return this value immediately.
    spare: ?f32 = null,

    /// Create an Rng seeded from the given u64.
    ///
    /// Worked example:
    ///   var rng = Rng.init(1337);  // deterministic from seed 1337
    ///   _ = rng.floatf32();        // first value is fixed for seed 1337
    ///
    /// Memory: no allocation.
    pub fn init(seed: u64) Rng {
        return .{
            // Xoshiro256.init takes a u64 and internally mixes it into the
            // 256-bit state.  The same seed always produces the same stream.
            .xoshiro = std.Random.Xoshiro256.init(seed),
        };
    }

    /// Returns a `std.Random` interface backed by this generator.
    ///
    /// The returned Random holds a pointer into `self.xoshiro`, so the Rng
    /// must outlive any use of the returned Random value.
    ///
    /// Worked example:
    ///   var rng = Rng.init(0);
    ///   const rand = rng.random();
    ///   const n = rand.int(u32);  // uniform u32
    ///
    /// Memory: no allocation.  The Random value borrows a pointer to self.
    pub fn random(self: *Rng) std.Random {
        // We delegate to Xoshiro256's own `.random()` method, which sets up
        // the vtable/fill pointer that std.Random uses under the hood.
        return self.xoshiro.random();
    }

    /// Returns a uniform f32 in [0, 1).
    ///
    /// Worked example:
    ///   var rng = Rng.init(42);
    ///   const x = rng.floatf32();  // 0.0 <= x < 1.0
    ///
    /// Memory: no allocation.
    pub fn floatf32(self: *Rng) f32 {
        // We go through our own `random()` so that all calls are consistent.
        // The std.Random.float(f32) method maps the generator's raw bits to
        // a uniform float in [0, 1) using the upper 24 bits (the f32 mantissa
        // has 23 explicit bits + 1 implicit leading 1 = 24 bits of precision).
        return self.random().float(f32);
    }

    /// Returns an f32 drawn from the standard normal distribution N(0, 1)
    /// using the Box-Muller transform.
    ///
    /// Every *other* call is "free" — the Box-Muller method naturally produces
    /// two independent normals (z0, z1); we return z0 immediately and cache
    /// z1 in `self.spare` for the next call.
    ///
    /// Worked example:
    ///   var rng = Rng.init(42);
    ///   const z = rng.normalF32();  // z ~ N(0, 1), e.g. -0.543...
    ///   // Internally, the second normal of the pair is cached for next call.
    ///
    /// Memory: no allocation.
    pub fn normalF32(self: *Rng) f32 {
        // Fast path: return the cached spare from the previous Box-Muller pair.
        if (self.spare) |s| {
            self.spare = null;
            return s;
        }

        // --- Box-Muller transform ---
        // uniform1 ∈ (0, 1]  — we use 1.0 - floatf32() to avoid the open end
        //                      at 0, which would cause ln(0) = -inf.
        // uniform2 ∈ [0, 1)  — standard uniform.
        const uniform1 = 1.0 - self.floatf32();
        const uniform2 = self.floatf32();

        // Common sub-expression: r = sqrt(-2 * ln(u1))
        // Since uniform1 > 0, ln(uniform1) is finite (though for very small
        // uniform1, the result can be large — this is correct for the normal
        // tail).
        const r = @sqrt(-2.0 * @log(uniform1));

        // Angle: theta = 2 * π * u2
        const theta = 2.0 * @as(f32, @floatCast(std.math.pi)) * uniform2;

        // Two independent normals from the same pair
        const z0 = r * @cos(theta);
        const z1 = r * @sin(theta);

        // Cache the second one for the next call
        self.spare = z1;

        return z0;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Rng.init produces deterministic sequence" {
    // Same seed → same sequence.  This is the core of D14.
    var rng_a = Rng.init(12345);
    var rng_b = Rng.init(12345);

    // Generate a handful of values and check they match exactly.
    for (0..10) |_| {
        const a = rng_a.floatf32();
        const b = rng_b.floatf32();
        try std.testing.expectEqual(a, b);
    }
}

test "Rng.floatf32 returns values in [0, 1)" {
    var rng = Rng.init(42);

    // Sample 1000 values and check they all fall in [0, 1).
    for (0..1000) |_| {
        const x = rng.floatf32();
        try std.testing.expect(x >= 0.0);
        try std.testing.expect(x < 1.0);
    }
}

test "Rng.normalF32 produces finite values" {
    var rng = Rng.init(42);

    // Sample 1000 normals and check they are finite (no NaN, no Inf).
    // We do not check the distribution shape here — that's the oracle's job.
    for (0..1000) |_| {
        const z = rng.normalF32();
        try std.testing.expect(std.math.isFinite(z));
    }
}

test "Rng.normalF32 uses cached spare (deterministic across calls)" {
    // Two RNGs with the same seed must produce the same normal sequence,
    // which also tests that the spare-caching logic is deterministic.
    var rng_a = Rng.init(999);
    var rng_b = Rng.init(999);

    for (0..20) |_| {
        const a = rng_a.normalF32();
        const b = rng_b.normalF32();
        try std.testing.expectEqual(a, b);
    }
}

test "Rng.normalF32 different seeds produce different sequences" {
    // Sanity: different seeds should not produce identical streams.
    var rng_0 = Rng.init(0);
    var rng_1 = Rng.init(1);

    const a = rng_0.normalF32();
    const b = rng_1.normalF32();

    // This *could* fail in theory if two seeds happen to produce the same
    // first normal, but with Xoshiro256 the probability is astronomically low.
    try std.testing.expect(a != b);
}
