//!
//! zig-transformer-lab — Tape-based reverse-mode autograd engine
//!
//! Purpose:
//!   The Tape records operations performed on tensors during a forward
//!   pass, then traverses them in reverse to compute gradients via
//!   the chain rule. This is the Zig equivalent of PyTorch's dynamic
//!   computational graph.
//!
//!   Usage pattern (one tape per training step):
//!     var tape = Tape.init(allocator);
//!     defer tape.deinit();
//!
//!     // Forward pass: ops with tape=... record nodes automatically
//!     var logits = try ops.matmul(allocator, x, w, &tape);
//!     var loss = try ops.crossEntropy(allocator, logits, targets, &tape);
//!
//!     // Backward pass: computes gradients for all requires_grad tensors
//!     try tape.backward(&loss);
//!
//!     // Access gradients: x.grad, w.grad, etc.
//!     // Optimizer step uses the gradients, then:
//!     tape.zeroGrad(params);  // reset for next step
//!
//! Shape contract:
//!   The tape itself has no shape — it's a list of Nodes. The backward
//!   traversal produces gradient tensors whose shapes match the original
//!   tensors' shapes.
//!
//! Math:
//!   Reverse-mode automatic differentiation (backpropagation):
//!
//!   Given a computation graph G = (V, E) where V is the set of tensor
//!   nodes and E is the set of operations, backward() performs:
//!
//!   1. Topological sort of G from the loss node (children before parents).
//!   2. Walk in reverse topological order.
//!   3. For each node v with operation op and upstream gradient dL/dv:
//!      - Compute dL/d(parent_i) using the backward rule for op.
//!      - Accumulate: grad_map[parent_i] += dL/d(parent_i)
//!
//!   The accumulation (+=) is critical: if a tensor is used in multiple
//!   operations, its gradient is the SUM of contributions from each use.
//!   This is the multivariable chain rule.
//!
//! Memory ownership:
//!   - The tape OWNS the ArrayList of Nodes (freed in deinit).
//!   - The grad_map OWNS all gradient tensors it creates (freed in deinit).
//!   - The Nodes' SavedData contains BORROWED pointers to tensors from
//!     the forward pass — these must outlive the tape.
//!   - When retain_graph=false (default), intermediate gradient tensors
//!     are freed immediately after their children are processed. This
//!     bounds memory usage to O(depth) instead of O(width).
//!
//! Errors:
//!   OutOfMemory — allocation failure during backward (gradient tensors)
//!   NumericalError — NaN detected in gradient (optional, not yet impl'd)
//!
//! TODO:
//!   - future: NaN detection in gradient accumulation (debug mode)
//!   - future: gradient clipping hooks inside the tape (clipping
//!     currently lives in the Trainer; moving it here would let
//!     tape users opt in via config).
//!
//! Credits:
//!   The tape-based autograd design is inspired by micrograd (Andrej
//!   Karpathy) and PyTorch's autograd engine. No code copied.
//;

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const NodeId = @import("../tensor/tensor.zig").NodeId;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;
const shape_equals = @import("../tensor/shape.zig").equals;
const node_mod = @import("node.zig");
const OpKind = node_mod.OpKind;
const Node = node_mod.Node;
const SavedData = node_mod.SavedData;
const backward_mod = @import("backward.zig");
// PR-η.2: the tape DtoD-copies CUDA saved tensors into tape-owned
// DeviceBuffers so backward can read them even after the caller's
// source tensors have been freed. DeviceBuffer owns the CUdeviceptr
// and routes deinit through cuMemFree_v2.
const DeviceBuffer = @import("../backend/cuda/mem.zig").DeviceBuffer;
// PR-ι: device-aware seed allocation for backward. onesLike routes
// to a CUDA fill (cuMemsetD32_v2 with the 1.0 bit pattern) when the
// loss tensor lives on GPU.
const ops_create = @import("../tensor/ops/create.zig");

// ---------------------------------------------------------------------------
// Tape — the autograd engine
// ---------------------------------------------------------------------------

/// The tape records operations and computes gradients.
///
/// Lifecycle:
///   1. init() — create an empty tape
///   2. Forward pass — ops call tape.record() when requires_grad
///   3. backward(loss) — compute all gradients
///   4. Access gradients via tensor.grad pointers
///   5. deinit() — free all tape-owned memory (nodes + gradient tensors)
///
/// One tape per training step is the intended usage. After backward()
/// and the optimizer step, deinit the tape and create a fresh one.
///
/// The tape does NOT own the forward-pass tensors. It only owns:
///   - The list of recorded Nodes
///   - The gradient tensors it creates during backward()
pub const Tape = struct {
    /// All nodes recorded during the forward pass, in execution order.
    /// Node IDs are indices into this list.
    nodes: std.ArrayList(Node),

    /// Maps NodeId → gradient tensor for that node's output.
    /// During backward, we look up each node's output gradient here,
    /// compute the input gradients, and accumulate them into the
    /// parents' entries.
    ///
    /// The tape owns all gradient tensors in this map — they are freed
    /// in deinit() (or earlier if retain_graph=false and the node is
    /// an intermediate that all children have been processed).
    grad_map: std.AutoHashMap(NodeId, *Tensor),

    /// Maps leaf NodeId → pointer to the original leaf tensor.
    /// After backward, we write gradients from grad_map back to
    /// tensor.grad for each leaf. Leaf tensors have tape_node=null
    /// initially — trackLeaf() assigns them a phantom node ID.
    leaf_map: std.AutoHashMap(NodeId, *Tensor),

    /// List of intermediate tensor data buffers that the tape keeps
    /// alive to prevent use-after-free during backward.
    ///
    /// SavedData stores tensor snapshots by value, including the
    /// `data` slice which points to the original heap buffer. If the
    /// original tensor is freed before backward runs, the slice dangles.
    /// By "donating" the data buffer to the tape, the buffer stays alive
    /// until tape.deinit() — the tape takes ownership.
    kept_alive: std.ArrayList([]f32),

    /// CUDA counterpart of `kept_alive`: DeviceBuffers owned by the
    /// tape, freed in `deinit` via `DeviceBuffer.deinit` (which calls
    /// cuMemFree_v2). Every CUDA tensor snapshot stored in SavedData
    /// points at one of these tape-owned buffers; caller tensors can
    /// be freed immediately after the forward op returns, same
    /// contract as the CPU path.
    kept_alive_cuda: std.ArrayList(DeviceBuffer),

    /// The allocator used for all tape allocations (nodes, gradient tensors).
    allocator: std.mem.Allocator,

    /// Next available NodeId. Incremented by record().
    next_id: NodeId,

    /// Whether to keep intermediate gradient tensors and graph structure
    /// after backward. Default false — frees memory eagerly.
    retain_graph: bool,

    /// Whether backward() has been called on this tape.
    /// Calling backward() twice without retain_graph is an error
    /// because intermediate data has been freed.
    backward_called: bool,

    /// Create a new empty tape.
    ///
    /// The tape uses the given allocator for all internal allocations
    /// (node list, gradient tensors). The caller must call deinit()
    /// when done to free all tape-owned memory.
    ///
    /// Worked example:
    ///   var tape = Tape.init(gpa);
    ///   defer tape.deinit();
    ///   // ... forward + backward ...
    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{
            .nodes = .empty,
            .grad_map = std.AutoHashMap(NodeId, *Tensor).init(allocator),
            .leaf_map = std.AutoHashMap(NodeId, *Tensor).init(allocator),
            .kept_alive = .empty,
            .kept_alive_cuda = .empty,
            .allocator = allocator,
            .next_id = 0,
            .retain_graph = false,
            .backward_called = false,
        };
    }

    /// Free all tape-owned memory.
    ///
    /// This frees:
    ///   - The node list
    ///   - All gradient tensors in grad_map
    ///
    /// It does NOT free the forward-pass tensors (inputs, intermediates,
    /// outputs) — those are owned by whoever created them.
    ///
    /// After deinit, the tape is in an undefined state.
    pub fn deinit(self: *Tape) void {
        // Before freeing the gradient tensors, null out every leaf's
        // `grad` pointer — otherwise the parameter tensors outlive
        // the tape with dangling pointers into freed gradients. The
        // next training step's forward path reads `.grad` (via
        // Tensor.transpose2d's view construction inheriting the
        // parent's grad pointer, then checkInvariants dereferencing
        // it) and crashes on freed memory.
        //
        // AdamW.zeroGrad could zero the grads, but that still leaves
        // them allocated (a large-ish overhead per param per step).
        // Clearing the leaf's pointer is a clean O(n_leaves) pass
        // that matches the "tape owns gradient tensors" contract.
        var leaf_iter = self.leaf_map.iterator();
        while (leaf_iter.next()) |entry| {
            entry.value_ptr.*.grad = null;
        }

        // Free all gradient tensors we created during backward.
        var iter = self.grad_map.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.grad_map.deinit();
        self.leaf_map.deinit();
        // Free kept-alive data buffers (transferred from intermediate tensors)
        for (self.kept_alive.items) |buf| {
            self.allocator.free(buf);
        }
        self.kept_alive.deinit(self.allocator);
        // Free kept-alive CUDA buffers. DeviceBuffer.deinit only frees
        // when `owned` is true, and we always allocate our copies as
        // owning in cloneTensorData's CUDA branch.
        for (self.kept_alive_cuda.items) |*buf| {
            buf.deinit();
        }
        self.kept_alive_cuda.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
    }

    /// Register a leaf tensor so it can receive gradients from backward.
    ///
    /// Leaf tensors (those created by the user, not by operations) start
    /// with tape_node=null. Without a node ID, the backward pass has
    /// nowhere to accumulate their gradient. trackLeaf() creates a
    /// phantom node for the leaf and stores the mapping so backward()
    /// can later write the gradient pointer back to tensor.grad.
    ///
    /// Call this BEFORE any op that uses the tensor as an input.
    ///
    /// Worked example:
    ///   var w = try Tensor.init(allocator, Shape.init2D(3, 4));
    ///   w.requires_grad = true;
    ///   _ = try tape.trackLeaf(&w);
    ///   // Now w.tape_node is set, and backward will populate w.grad
    pub fn trackLeaf(self: *Tape, tensor: *Tensor) LabError!NodeId {
        // Always create a fresh leaf node for this tape, even if the
        // tensor already has a tape_node from a previous step's tape.
        // The previous tape has been deinited, so the old tape_node
        // is stale — it references a node in a destroyed tape's node
        // list. If we skip re-registration, the new tape's intermediate
        // nodes will collide with the old tape_node IDs, causing
        // gradient accumulation between unrelated tensors (shape mismatch).
        const id = self.next_id;
        self.next_id += 1;

        // Create a placeholder leaf node with zero parents.
        // The backward pass will skip it (n_parents=0), but its
        // gradient slot in grad_map will be populated by child nodes.
        const leaf_node = Node{
            .id = id,
            .op = undefined,
            .parents = .{ null, null },
            .n_parents = 0,
            .saved = .nothing,
        };
        try self.nodes.append(self.allocator, leaf_node);

        // Link the leaf tensor to this node ID
        tensor.tape_node = id;

        // Remember where to write the gradient after backward
        try self.leaf_map.put(id, tensor);

        return id;
    }

    /// Record an operation on the tape and return its NodeId.
    ///
    /// This is called by the autograd-enabled ops (in the various
    /// tensor/ops/*.zig files) when they detect that any input has
    /// requires_grad=true.
    ///
    /// The caller fills in the Node fields (op, parents, n_parents, saved)
    /// and this function assigns the ID and appends it to the node list.
    ///
    /// Worked example:
    ///   // Inside autograd-enabled add():
    ///   const node = Node{
    ///       .id = undefined,  // will be assigned by record()
    ///       .op = .add,
    ///       .parents = .{ a.tape_node, b.tape_node },
    ///       .n_parents = 2,
    ///       .saved = .nothing,
    ///   };
    ///   const node_id = try tape.record(node);
    ///   // node_id is the ID; store it on the output tensor:
    ///   result.tape_node = node_id;
    pub fn record(self: *Tape, node: Node) LabError!NodeId {
        const id = self.next_id;
        self.next_id += 1;

        // PR-ε: the tape takes ownership of any buffer referenced by
        // `node.saved`. Before this change, SavedData stored Tensor
        // snapshots whose `.data` slice pointed at the caller's heap
        // buffer; if the caller freed its tensor before backward (as
        // happens for every intermediate produced inside an nn/ layer
        // forward method) the saved slice dangled. The previous
        // workaround was for layer code to call `tape.keepAlive(&t)`
        // on every intermediate — a distributed, easy-to-forget
        // contract that read incorrectly as "the layer author owns the
        // tape's memory". The fix is local to record(): we walk the
        // SavedData union, copy any referenced buffers into
        // tape-owned memory, and rewrite the Tensor snapshots to point
        // at those copies. Backward reads identically; the caller's
        // tensors can be freed immediately after the op returns.
        const owned_saved = try self.takeOwnershipOfSaved(node.saved);

        const full_node = Node{
            .id = id,
            .op = node.op,
            .parents = node.parents,
            .n_parents = node.n_parents,
            .saved = owned_saved,
        };

        self.nodes.append(self.allocator, full_node) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
        };

        return id;
    }

    /// Walk a SavedData value, allocate tape-owned copies of every
    /// referenced buffer, and return a new SavedData whose tensor
    /// snapshots point at those copies. The copies are tracked in
    /// `self.kept_alive` so `Tape.deinit` can free them in one pass.
    fn takeOwnershipOfSaved(self: *Tape, saved: SavedData) LabError!SavedData {
        return switch (saved) {
            // Variants that reference no caller-owned buffers pass
            // through unchanged.
            .nothing, .tensor_scalar, .reduce_info => saved,

            .tensor_ref => |t| SavedData{
                .tensor_ref = try self.cloneTensorData(t),
            },
            .tensor_pair => |p| SavedData{
                .tensor_pair = .{
                    .a = try self.cloneTensorData(p.a),
                    .b = try self.cloneTensorData(p.b),
                },
            },
            .ce_info => |info| SavedData{
                .ce_info = .{
                    .logits = try self.cloneTensorData(info.logits),
                    .targets = try self.cloneSlice(info.targets),
                },
            },
            .ce_cuda_grad => |t| SavedData{
                // The fused CUDA forward already computed the gradient.
                // cloneTensorData DtoD-copies it into kept_alive_cuda
                // so backward can read it after the caller's grad
                // tensor has been freed.
                .ce_cuda_grad = try self.cloneTensorData(t),
            },
            .embedding_info => |info| SavedData{
                .embedding_info = .{
                    .weight = try self.cloneTensorData(info.weight),
                    .indices = try self.cloneSlice(info.indices),
                },
            },
        };
    }

    /// Allocate a tape-owned copy of `t.data` and return a Tensor
    /// snapshot whose `.data` points at the copy. Shape, strides,
    /// and device are preserved verbatim. The returned Tensor's
    /// `owned` / `storage.cpu.owned` are set to false — the tape
    /// owns the buffer, not this snapshot.
    ///
    /// CUDA branch (PR-η.2): for a CUDA tensor we allocate a new
    /// DeviceBuffer sized to the source buffer, DtoD-copy, track it
    /// in `kept_alive_cuda`, and return a snapshot whose
    /// `storage.cuda` is a non-owning view of the tape-owned buffer.
    /// The top-level `owned` compat alias stays false (PR-δ
    /// invariant for CUDA tensors).
    fn cloneTensorData(self: *Tape, t: Tensor) LabError!Tensor {
        return switch (t.storage) {
            .cpu => |s| blk: {
                const copy = self.allocator.dupe(f32, s.data) catch return error.OutOfMemory;
                self.kept_alive.append(self.allocator, copy) catch {
                    self.allocator.free(copy);
                    return error.OutOfMemory;
                };
                var c = t;
                // The snapshot borrows the buffer (tape owns it via kept_alive).
                c.storage = .{ .cpu = .{ .data = copy, .owned = false } };
                break :blk c;
            },
            .cuda => |src_buf| blk: {
                // Allocate a fresh owning DeviceBuffer on the same
                // context the source lives on, DtoD copy its bytes,
                // and transfer ownership into kept_alive_cuda.
                var tape_buf = try DeviceBuffer.alloc(src_buf.ctx, src_buf.len);
                errdefer tape_buf.deinit();
                try tape_buf.copyFromDevice(src_buf);
                self.kept_alive_cuda.append(self.allocator, tape_buf) catch {
                    tape_buf.deinit();
                    return error.OutOfMemory;
                };
                // The snapshot returned to SavedData must be
                // non-owning (the tape owns the buffer). We recover
                // the stored pointer/len/ctx and set owned=false.
                const stored = self.kept_alive_cuda.items[self.kept_alive_cuda.items.len - 1];
                var c = t;
                c.storage = .{ .cuda = .{
                    .ctx = stored.ctx,
                    .ptr = stored.ptr,
                    .len = stored.len,
                    .owned = false,
                } };
                break :blk c;
            },
        };
    }

    /// Allocate a tape-owned copy of a `[]const f32` (used by ce_info
    /// for targets and embedding_info for indices). Tracked in
    /// `kept_alive` so deinit frees it.
    fn cloneSlice(self: *Tape, src: []const f32) LabError![]const f32 {
        const copy = self.allocator.dupe(f32, src) catch return error.OutOfMemory;
        self.kept_alive.append(self.allocator, copy) catch {
            self.allocator.free(copy);
            return error.OutOfMemory;
        };
        return copy;
    }

    /// Transfer ownership of a tensor's data buffer to the tape.
    ///
    /// DEPRECATED (PR-ε): `tape.record()` now copies saved buffers
    /// automatically, so layer code no longer needs to call this.
    /// It is kept as a no-op for one transition PR so old call sites
    /// compile while they are removed; the function will be deleted
    /// entirely once `nn/` and `gradcheck` are clean.
    ///
    /// The old contract (transfer the tensor's buffer to the tape and
    /// mark the tensor non-owning) is a subtle footgun: callers had
    /// to manually reason about which intermediates would be touched
    /// by backward and arrange for their buffers to outlive the tape.
    /// The new design — tape owns copies of only what it needs — is
    /// what PyTorch's `ctx.save_for_backward` does internally.
    pub fn keepAlive(self: *Tape, tensor: *Tensor) !void {
        _ = self;
        _ = tensor;
        // Intentionally empty. See doc comment.
    }

    /// Run the backward pass from the given loss tensor.
    ///
    /// This computes dL/d(input) for every tensor that has
    /// requires_grad=true and was part of the computation graph
    /// that produced `loss`.
    ///
    /// Algorithm:
    ///   1. Seed the gradient of the loss node with 1.0 (a scalar all-ones).
    ///   2. Topological sort: walk backward from the loss, collecting
    ///      all nodes reachable from it.
    ///   3. Walk in reverse topological order (children before parents).
    ///   4. For each node, look up its output gradient, compute the
    ///      input gradients via the op-specific backward function,
    ///      and accumulate them into the parents' grad_map entries.
    ///   5. After processing each intermediate node (non-leaf), if
    ///      retain_graph=false, free its gradient tensor.
    ///
    /// After backward(), each leaf tensor with requires_grad=true will
    /// have its .grad pointer set to the computed gradient tensor.
    ///
    /// Worked example:
    ///   // After forward: loss = (a * b + c)^2
    ///   try tape.backward(&loss);
    ///   // Now: a.grad, b.grad, c.grad contain the computed gradients
    pub fn backward(self: *Tape, loss: *Tensor) LabError!void {
        std.debug.assert(loss.tape_node != null or loss.requires_grad);

        // Seed: the gradient of the loss w.r.t. itself is 1.0.
        // For a scalar loss (shape (1,)), this is just [1.0].
        // For a non-scalar loss, every element gets gradient 1.0
        // (this corresponds to the "vector-Jacobian product" with
        // the all-ones vector).
        const loss_id = loss.tape_node orelse {
            // Loss has no tape node — it's a leaf with requires_grad
            // but no operations were recorded. Nothing to backprop.
            // Create a gradient of 1.0 for it on the same device.
            const grad = try ops_create.onesLike(self.allocator, loss.*);
            const grad_ptr = try self.allocator.create(Tensor);
            grad_ptr.* = grad;
            try self.grad_map.put(loss.tape_node orelse 0, grad_ptr);
            loss.grad = grad_ptr;
            return;
        };

        // Create the seed gradient for the loss node on the same
        // device as the loss (PR-ι). Previously this was
        // Tensor.init + fill(1.0), which silently allocated on CPU
        // even when loss lived on CUDA.
        const seed = try ops_create.onesLike(self.allocator, loss.*);
        const seed_ptr = try self.allocator.create(Tensor);
        seed_ptr.* = seed;
        try self.grad_map.put(loss_id, seed_ptr);
        loss.grad = seed_ptr;

        // Step 1: Topological sort — collect all nodes reachable
        // from the loss node, then order them so children come
        // before parents.
        //
        // We do this with a simple DFS from the loss node, then
        // reverse the post-order to get the correct backward order.
        var topo_order = std.ArrayList(NodeId).empty;
        defer topo_order.deinit(self.allocator);

        var visited = std.AutoHashMap(NodeId, void).init(self.allocator);
        defer visited.deinit();

        try self.topologicalSort(loss_id, &topo_order, &visited);

        // Step 2: Walk in reverse topological order and compute
        // gradients for each node's parents.
        //
        // The topo_order is in "forward-ish" order (children first),
        // The topo_order is in forward-pass order (children after parents).
        // For backward we walk in REVERSE — output node first, then
        // its inputs — so each node's output gradient has already been
        // accumulated by the time we process it.
        var i: usize = topo_order.items.len;
        while (i > 0) {
            i -= 1;
            const node_id = topo_order.items[i];
            // Look up this node
            const node = self.nodes.items[node_id];

            // Leaf nodes (n_parents=0) have no backward to run;
            // their gradient is accumulated by child nodes.
            if (node.n_parents == 0) continue;

            // Get the output gradient for this node
            const grad_out_ptr = self.grad_map.get(node_id) orelse {
                // This can happen for nodes that are not on the path
                // from loss to leaves (e.g., a computation that
                // doesn't contribute to the loss). Skip them.
                continue;
            };

            // Compute input gradients via the op-specific backward
            const parent_grads = try backward_mod.backward(
                self.allocator,
                node,
                grad_out_ptr,
            );

            // Accumulate gradients into parents
            for (0..node.n_parents) |pi| {
                const parent_id = node.parents[pi] orelse continue;
                const parent_grad = parent_grads[pi] orelse continue;

                if (self.grad_map.get(parent_id)) |existing| {
                    backward_mod.accumulateGrad(existing, parent_grad);
                    // Free the temporary parent_grad since we've
                    // accumulated its values into existing
                    parent_grad.deinit(self.allocator);
                    self.allocator.destroy(parent_grad);
                } else {
                    // First time we've seen this parent — store it
                    try self.grad_map.put(parent_id, parent_grad);
                }
            }

            // Free any parent gradient tensors that weren't consumed
            // above (e.g., when a parent has no tape_node because it
            // doesn't require grad). Without this cleanup, those
            // tensors would leak — backward still computes them but
            // the accumulation loop skips null parent IDs.
            for (0..node.n_parents) |pi| {
                const parent_grad = parent_grads[pi] orelse continue;
                // If this parent's gradient was stored in grad_map,
                // it's now owned by grad_map — don't free it here.
                // We only free gradients whose parent_id was null
                // (skipped in the accumulation loop above).
                const parent_id = node.parents[pi] orelse {
                    // parent_id is null — this gradient was computed
                    // but never stored. Free it now.
                    parent_grad.deinit(self.allocator);
                    self.allocator.destroy(parent_grad);
                    continue;
                };
                // parent_id is non-null — check if it was stored or
                // accumulated. If stored, grad_map owns it. If
                // accumulated, it was already freed above. In either
                // case, we must NOT free it here.
                _ = parent_id;
            }

            // Free intermediate gradients if retain_graph=false.
            //
            // Every entry in self.nodes is a recorded op with ≥ 1 parent;
            // user-supplied leaf tensors reach the tape only via
            // trackLeaf(), which creates placeholder nodes with
            // n_parents=0. So `node.n_parents > 0` cleanly identifies
            // intermediates. We keep the loss's gradient (the seed) until
            // deinit so callers can still inspect loss.grad; everything
            // else can be freed once its parents have accumulated it.
            if (!self.retain_graph) {
                if (node.n_parents > 0 and node_id != loss_id) {
                    if (self.grad_map.fetchRemove(node_id)) |kv| {
                        kv.value.deinit(self.allocator);
                        self.allocator.destroy(kv.value);
                    }
                }
            }
        }

        // Step 3: Write gradient pointers back to leaf tensors.
        //
        // Leaf tensors (parameters, model inputs) enter the tape via
        // trackLeaf(), which assigns them a fresh NodeId and stores the
        // mapping (node_id → *Tensor) in self.leaf_map. After reverse
        // propagation has populated grad_map for every reachable node,
        // we walk leaf_map and copy the pointer from grad_map into
        // each leaf's .grad field. This is what lets callers write:
        //
        //     tape.backward(&loss);
        //     optimizer.step(params);   // reads param.grad
        //
        // without ever interacting with NodeIds themselves.
        // For each leaf registered via trackLeaf(), check if grad_map
        // has a gradient for its node ID and set tensor.grad.
        var leaf_iter = self.leaf_map.iterator();
        while (leaf_iter.next()) |entry| {
            const leaf_id = entry.key_ptr.*;
            const tensor_ptr = entry.value_ptr.*;
            if (self.grad_map.get(leaf_id)) |grad| {
                tensor_ptr.grad = grad;
            }
        }

        self.backward_called = true;
    }

    /// Topological sort via DFS from the given node.
    ///
    /// Produces a post-order traversal: children are visited before
    /// their parents. Walking the result in forward order gives us
    /// the correct backward pass sequence.
    fn topologicalSort(
        self: *Tape,
        start_id: NodeId,
        order: *std.ArrayList(NodeId),
        visited: *std.AutoHashMap(NodeId, void),
    ) LabError!void {
        if (visited.contains(start_id)) return;
        try visited.put(start_id, {});

        // If start_id is out of range (e.g., a leaf that was never
        // recorded as a node), skip it.
        if (start_id >= self.nodes.items.len) return;

        const node = self.nodes.items[start_id];

        // Visit all parents first (DFS on dependencies)
        for (0..node.n_parents) |i| {
            const parent_id = node.parents[i] orelse continue;
            try self.topologicalSort(parent_id, order, visited);
        }

        // After all parents are visited, add this node.
        // This gives us a valid topological order where
        // a node appears after all its dependencies.
        try order.append(self.allocator, start_id);
    }

    /// Reset all gradient tensors to zero for the given parameter list.
    ///
    /// This is called after the optimizer step to prepare for the next
    /// forward pass. It does NOT free the gradient tensors — it just
    /// fills them with zeros so they're ready for the next accumulation.
    ///
    /// Worked example:
    ///   // After optimizer.step():
    ///   tape.zeroGrad(&params);
    ///   // Now all params[i].grad tensors contain zeros
    pub fn zeroGrad(_: *Tape, params: []const *Tensor) void {
        for (params) |p| {
            if (p.grad) |g| {
                g.fill(0.0);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const create = @import("../tensor/ops/create.zig");

test "Tape init/deinit — no leak" {
    var tape = Tape.init(std.testing.allocator);
    defer tape.deinit();
    // Just verify it creates and destroys cleanly
    try std.testing.expectEqual(@as(NodeId, 0), tape.next_id);
    try std.testing.expectEqual(@as(usize, 0), tape.nodes.items.len);
}

test "Tape record returns incrementing IDs" {
    var tape = Tape.init(std.testing.allocator);
    defer tape.deinit();

    const id0 = try tape.record(Node{
        .id = undefined,
        .op = .add,
        .parents = .{ null, null },
        .n_parents = 0,
        .saved = .nothing,
    });
    const id1 = try tape.record(Node{
        .id = undefined,
        .op = .mul,
        .parents = .{ id0, null },
        .n_parents = 1,
        .saved = .nothing,
    });

    try std.testing.expectEqual(@as(NodeId, 0), id0);
    try std.testing.expectEqual(@as(NodeId, 1), id1);
    try std.testing.expectEqual(@as(usize, 2), tape.nodes.items.len);

    // Verify the stored nodes have the right IDs
    try std.testing.expectEqual(@as(NodeId, 0), tape.nodes.items[0].id);
    try std.testing.expectEqual(@as(NodeId, 1), tape.nodes.items[1].id);
}

test "Tape topologicalSort — simple chain" {
    var tape = Tape.init(std.testing.allocator);
    defer tape.deinit();

    // Build: n0 → n1 → n2 (n0 is the leaf, n2 is the output)
    const id0 = try tape.record(Node{
        .id = undefined,
        .op = .relu,
        .parents = .{ null, null },
        .n_parents = 0,
        .saved = .nothing,
    });
    const id1 = try tape.record(Node{
        .id = undefined,
        .op = .relu,
        .parents = .{ id0, null },
        .n_parents = 1,
        .saved = .nothing,
    });
    _ = try tape.record(Node{
        .id = undefined,
        .op = .relu,
        .parents = .{ id1, null },
        .n_parents = 1,
        .saved = .nothing,
    });

    var order = std.ArrayList(NodeId).empty;
    defer order.deinit(std.testing.allocator);
    var visited = std.AutoHashMap(NodeId, void).init(std.testing.allocator);
    defer visited.deinit();

    try tape.topologicalSort(id0, &order, &visited);

    // id0 has no parents, so the order should just be [id0]
    try std.testing.expectEqual(@as(usize, 1), order.items.len);
    try std.testing.expectEqual(@as(NodeId, id0), order.items[0]);
}

test "Tape topologicalSort — two-step chain" {
    var tape = Tape.init(std.testing.allocator);
    defer tape.deinit();

    // n0 (leaf) → n1 (relu)
    const id0 = try tape.record(Node{
        .id = undefined,
        .op = .relu,
        .parents = .{ null, null },
        .n_parents = 0,
        .saved = .nothing,
    });
    const id1 = try tape.record(Node{
        .id = undefined,
        .op = .relu,
        .parents = .{ id0, null },
        .n_parents = 1,
        .saved = .nothing,
    });

    var order = std.ArrayList(NodeId).empty;
    defer order.deinit(std.testing.allocator);
    var visited = std.AutoHashMap(NodeId, void).init(std.testing.allocator);
    defer visited.deinit();

    try tape.topologicalSort(id1, &order, &visited);

    // id1 depends on id0, so topo order is: [id0, id1]
    try std.testing.expectEqual(@as(usize, 2), order.items.len);
    try std.testing.expectEqual(@as(NodeId, id0), order.items[0]);
    try std.testing.expectEqual(@as(NodeId, id1), order.items[1]);
}
