//! docs_linecount — cross-platform replacement for `wc -l docs/0*.md`.
//!
//! Walks the `docs/` directory, finds every file whose name starts with
//! `0` and ends in `.md`, counts its lines, and prints one line per file
//! plus a total. Used by `zig build docs` to verify that each chapter
//! meets the 500-line minimum set in the project's docs plan.
//!
//! Why a tiny helper executable instead of `sh -c`?
//!   - `sh -c` doesn't exist on Windows by default.
//!   - `zig build` on Windows used to fail at this step.
//!   - A Zig program is portable and doesn't need external tools.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    // Arena lifetime matches the whole program — simpler than tracking
    // individual allocations for a short-lived helper.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const io = init.io;
    const cwd = std.Io.Dir.cwd();

    // Open docs/ for iteration. Missing directory is a warning, not an error —
    // this step is informational.
    var dir = cwd.openDir(io, "docs", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => {
            var buf: [0]u8 = undefined;
            const locked = init.io.lockStderr(&buf, null) catch return;
            _ = locked.file_writer.interface.writeAll(
                "docs/ directory not found; nothing to count\n",
            ) catch {};
            return;
        },
        else => return err,
    };
    defer dir.close(io);

    // Collect matching filenames, then sort for deterministic output.
    var names: std.ArrayList([]u8) = .empty;
    defer names.deinit(allocator);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (entry.name.len < 4) continue;
        if (entry.name[0] != '0') continue;
        if (!std.mem.endsWith(u8, entry.name, ".md")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }

    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    // Use locked stderr like every other example in this project.
    var out_buf: [0]u8 = undefined;
    const locked = init.io.lockStderr(&out_buf, null) catch return;
    const writer = &locked.file_writer.interface;

    try writer.writeAll("=== docs/ line counts ===\n");
    var total: usize = 0;
    for (names.items) |name| {
        const count = try countLines(io, dir, name, allocator);
        total += count;
        try writer.print("{d:>6} docs/{s}\n", .{ count, name });
    }
    try writer.print("{d:>6} total\n", .{total});
}

fn countLines(io: std.Io, dir: std.Io.Dir, name: []const u8, allocator: std.mem.Allocator) !usize {
    // Read the whole file (cap at 16 MiB — largest doc today is ~1700 lines).
    const bytes = dir.readFileAlloc(io, name, allocator, .limited(16 * 1024 * 1024)) catch {
        return 0;
    };
    defer allocator.free(bytes);
    var count: usize = 0;
    for (bytes) |b| {
        if (b == '\n') count += 1;
    }
    return count;
}
