# MLX Backend Remediation Roadmap

**Date:** 2026-03-20  
**Status:** Proposed remediation plan based on static review

## Goal

Bring the MLX backend to a state where:

- memory behavior is predictable and safe under training workloads,
- large-model limits are clearly separated into true MLX/Metal constraints vs backend design constraints,
- the backend exposes the right primitives for memory-efficient MLX training,
- and training code does not silently fall onto unsupported or pathological execution paths.

This plan is written specifically with the fused chunker training failures in mind, but the recommendations apply to large-model training generally.

## Executive Summary

The current MLX backend appears to have fixed the major leak and lifecycle bugs documented in the downstream training notes. The remaining problem is less about leaks and more about **execution strategy**.

At present:

- MLX bridge and executable memory management are much better,
- but large-model backward execution still relies on explicit tensor materialization,
- donation is ignored,
- MLX checkpointing and native autograd transforms are exposed in the bridge but not integrated into the backend execution strategy,
- and several operations still silently force the backend out of the pure C-tape path and into slower, riskier Go callback execution.

The main conclusion is:

**The 14.3 GB failure mode looks mostly real for the current execution design, but not necessarily fundamental to MLX as a platform.**

If MLX is to support larger training workloads better, the path forward is:

1. stronger memory safety controls,
2. explicit fallback detection,
3. input donation,
4. checkpointed MLX-native autograd/blockwise training,
5. and clearer backend capability contracts.

## Non-Negotiable Acceptance Criteria

Before declaring MLX “ready” for large-model training, require:

1. no silent Go-callback fallback on the intended training graph,
2. explicit input donation support or equivalent ownership transfer semantics,
3. one checkpointed/autograd training path integrated and benchmarked,
4. memory safety limits and early-fail guards in place,
5. backend capability reporting that distinguishes full support from partial/slow support,
6. and reproducible evidence that large-model failures are due to a true tensor-size wall rather than avoidable backend overhead.

## Current Diagnosis

## What Looks Fixed

Based on the code and the downstream progress notes, the following issues appear to have been addressed:

- leaked output/result handle references in the C tape interpreter,
- leaked intermediate arrays in Go replay closure execution,
- `Array.Free()` allocating replacement empty handles,
- `mlx_compile`-related growth on repeated closure use,
- and several training-loop result/input finalization gaps downstream.

These fixes should be preserved and treated as regression-sensitive.

## What Still Looks Structurally Weak

### 1. The backend does not yet use MLX’s main memory-saving features

The bridge exposes:

- `ValueAndGrad`
- `VJP`
- `JVP`
- `Checkpoint`

But the backend execution layer only treats closures as forward executables. The current system therefore leaves the most important memory/computation tradeoff primitive unused.

### 2. Donation is ignored

Both executable paths accept `donate []bool`, but neither actually uses it. That means large ephemeral training inputs survive longer than they need to.

### 3. Some ops still kick the backend out of the “safe” execution path

Several operations call `markGoCallback()` and therefore force fallback to raw Go replay closure execution. Even if this is functionally correct, it is a dangerous performance and memory cliff for large training graphs.

Examples include:

- RNG / dropout path
- dynamic slice / dynamic update slice
- batch norm training/inference compound ops
- some compound ops that are still only encoded as Go replay

### 4. Capability declarations are too optimistic for training planning

The backend currently advertises support for several ops whose support is:

- partial,
- limited to simple cases,
- or functionally correct but operationally unsuitable for large training graphs.

### 5. Safety controls are still soft

The backend sets advisory MLX memory/cache limits, but the bridge also exposes a wired-limit hook that is not used. There is no backend-level early-fail strategy when memory approaches a dangerous regime.

### 6. The remaining 14.3 GB problem is likely execution-design-bound

The current MLX design still constructs and materializes explicit intermediate arrays for large operations. Unlike MPSGraph, it does not have a compiler layer here that transparently tiles/streams giant backward computations across smaller internal buffers.

That means:

- for the current segmented synthetic-VJP style,
- and for current `DotGeneral` decomposition/materialization behavior,

the Metal buffer cap is a real limit.

## Workstreams

## 1. Add Hard Memory Safety Controls

### Problem

`SetMemoryLimit` and `SetCacheLimit` are advisory. They help cache behavior, but they do not guarantee process safety. The bridge exposes stricter memory-related controls, but the backend does not yet use them.

### Remediation

Add a safety layer that treats memory governance as a first-class backend concern.

### Implementation Plan

1. Make backend memory fractions configurable rather than fixed constants.
2. Add support for bridge-level wired-memory limit configuration.
3. Expose runtime memory guardrails:
   - active memory threshold,
   - peak memory threshold,
   - optional fail-fast threshold.
4. Allow training code to query a “safe budget” from the backend.

### Deliverables

- configurable memory safety policy in `mlx/backend.go`
- optional hard-stop behavior before system instability
- memory policy startup summary in logs

### Exit Criteria

- large MLX training runs can fail early and safely instead of drifting into OS-level instability.

## 2. Implement Real Donation Semantics

### Problem

The executable paths ignore `donate []bool`. That leaves avoidable buffer lifetime pressure in training loops.

### Remediation

Honor input donation when the caller indicates ownership can be transferred.

### Implementation Plan

1. Define exact semantics for donated `mlxBuffer` inputs:
   - after successful execution, donated buffers are invalidated/finalized,
   - non-donated buffers remain caller-owned.
2. Ensure this is safe with:
   - shared-buffer inputs,
   - output aliasing assumptions,
   - control-flow executables.
3. Add explicit validation so donated buffers cannot be reused accidentally.

### Deliverables

- donation-aware `Execute` for standard executable
- donation-aware `Execute` for control-flow executable
- tests for correct invalidation and no double-free

### Exit Criteria

- training loops can release ephemeral combined-input buffers immediately through the backend rather than relying on external finalization timing.

## 3. Expose Checkpointing and Native MLX Autograd Upstack

### Problem

The bridge already has `Checkpoint`, `ValueAndGrad`, and `VJP`, but the backend does not turn them into a usable training strategy.

### Remediation

Make MLX-native differentiation and checkpointing a supported backend/training mode rather than an unused bridge feature.

### Design Goal

Support:

- forward closure,
- checkpointed forward closure,
- value-and-grad closure,
- optional blockwise checkpointed value-and-grad closure.

### Implementation Plan

1. Add an internal execution abstraction for:
   - forward-only closures,
   - checkpointed closures,
   - MLX-native gradient closures.
2. Add backend extensions or trainer-visible interfaces so training code can request:
   - `CompileForward`
   - `CompileValueAndGrad`
   - `CompileCheckpointedValueAndGrad`
3. Prefer MLX-native autograd for MLX-specific training paths instead of building high-memory synthetic-VJP graphs externally.

### Deliverables

- backend-level pathway for compiled MLX-native differentiation
- explicit training-mode flag for checkpoint/rematerialization

### Exit Criteria

- there is at least one supported MLX training path that uses checkpointing to trade compute for memory.

## 4. Move MLX Away from the MPSGraph Segmented-VJP Strategy

### Problem

The current large-model failure is strongly tied to a segmented synthetic-VJP style that was designed around MPSGraph constraints, not MLX strengths.

### Remediation

Treat MLX and MPSGraph as different backend strategies.

### Recommendation

For MLX:

- avoid MPSGraph-style segmented synthetic VJP as the primary large-model strategy,
- prefer monolithic or blockwise closures with MLX-native reverse-mode autograd,
- add checkpointing and microbatching,
- use gradient accumulation instead of relying on large instantaneous backward tensors.

For MPSGraph:

- continue using segmented execution where that backend benefits from it.

### Deliverables

- documented MLX-specific training strategy
- explicit backend routing policy in downstream training code

### Exit Criteria

- MLX is no longer forced through an execution strategy that is known to create giant explicit backward tensors.

## 5. Make Go-Callback Fallback Explicit

### Problem

The backend currently allows graphs to silently cross from the pure C-tape path into Go callback execution when certain ops appear.

This is dangerous because:

- performance changes abruptly,
- memory behavior changes,
- and backend suitability for large training can silently collapse.

### Remediation

Introduce explicit execution-path classification.

### Proposed Execution Classes

- `ctape_fast` — pure C tape path, suitable for large workloads
- `compiled_go_closure` — compiled Go closure, acceptable but less ideal
- `raw_go_closure` — functional fallback, not recommended for large training

### Implementation Plan

1. Add builder/executable metadata describing which path was selected.
2. Add optional strict mode:
   - fail if `raw_go_closure` is selected for training.
3. Log which ops triggered fallback.

### Deliverables

- execution path introspection
- strict mode for training backends
- fallback-op reporting

### Exit Criteria

- no training run uses a fallback path without making that fact obvious.

## 6. Tighten Capability Contracts

### Problem

The backend capability table says many ops are supported, but some are:

- simple-case only,
- Go-callback only,
- or not appropriate for large training graphs.

### Remediation

Separate “functionally implemented” from “large-train-safe.”

### Proposed Capability Model

For each op, track:

- `supported`
- `ctape_supported`
- `general_case_supported`
- `train_safe`

This can be metadata only at first.

### Deliverables

- clearer internal capability classification
- reduced ambiguity when new model architectures are brought onto MLX

### Exit Criteria

- downstream code can make informed routing decisions instead of assuming all `true` capabilities are equally viable.

## 7. Audit Remaining Lifecycle and Stream Ownership Gaps

### Problem

Most major lifecycle bugs appear fixed, but some small patterns still look risky.

One example is zero-initialization paths that allocate default streams inline.

### Remediation

Perform a focused lifecycle audit of:

- array creation helpers,
- default stream creation/freeing,
- buffer wrapping/unwrapping,
- `DataPtr` usage assumptions,
- closure ownership and payload lifetime,
- and any helper that bypasses normal stream ownership rules.

### High-Priority Audit Targets

- `NewArrayFromData(nil, ...)`
- `DefaultGPUStream()` / `DefaultCPUStream()` usage in utility helpers
- any place that evaluates and reads `DataPtr()` without a clear lifetime boundary

### Deliverables

- lifecycle audit checklist
- regression tests for stream/array helper safety

### Exit Criteria

- helper-level resource behavior is boring and unsurprising under stress.

## 8. Improve `DotGeneral` Strategy for Large Backward Graphs

### Problem

General `DotGeneral` is decomposed into explicit transpose + reshape + matmul + reshape with explicit temp slots. This is correct, but for large training graphs it still encourages explicit giant intermediates.

### Remediation

Revisit how MLX large matmul-heavy graphs are represented.

### Options

1. Keep current decomposition for correctness, but:
   - prefer blockwise/training-specific rewrites upstream,
   - use checkpointing to avoid storing intermediates.

2. Add specialized MLX-side kernels/paths for more `DotGeneral` forms if mlx-c exposes them.

3. Introduce backend-level heuristics that:
   - detect dangerous tensor shapes,
   - reject them early,
   - or route them to alternative strategies.

### Deliverables

- shape-risk heuristics for large dot products
- documented “safe zone” for MLX training configs

### Exit Criteria

- large tensor failures are predicted and surfaced before a training run reaches them.

## 9. Add a Large-Model Qualification Matrix

### Problem

Right now “MLX works” and “MLX works for this exact training regime” are getting conflated.

### Remediation

Define qualification tiers.

### Proposed Tiers

#### Tier A: Safe General Use

- inference
- small/medium training
- small VJP workloads

#### Tier B: Advanced Training

- checkpointed training
- microbatched training
- blockwise autograd

#### Tier C: Not Recommended Yet

- large segmented synthetic-VJP training with high batch and long sequence
- workloads known to exceed single-buffer limits

### Deliverables

- a backend qualification matrix by model size / sequence / batch / training mode

### Exit Criteria

- backend recommendations are precise and operationally useful.

## 10. Recommended Training Strategy for Large Models on MLX

This is the most important policy recommendation.

### Do Not Use as Primary Strategy

- MPSGraph-style segmented synthetic VJP for very large models
- dropout/RNG-heavy raw Go-callback graphs
- large-batch long-sequence backward passes without checkpointing

### Prefer Instead

1. MLX-native autograd closures
2. checkpointed blockwise execution
3. microbatching
4. gradient accumulation
5. explicit no-dropout / no-RNG training mode when needed

### Practical Positioning

- **MPSGraph** should remain the primary backend for current 22-layer fused chunker production training.
- **MLX** should be treated as:
  - excellent for smaller training jobs,
  - good for inference,
  - and a promising future large-training backend only after checkpointed/native-autograd integration lands.

## 11. Recommended Implementation Order

Do the work in this order.

### Stage 0: Documentation Reset

- update backend docs to distinguish:
  - memory-safe,
  - train-safe,
  - and large-model-capable.

### Stage 1: Safety and Transparency

- configurable memory policy
- wired-limit support
- execution-path reporting
- strict mode for Go-callback fallback

### Stage 2: Donation

- implement `donate[]` semantics in executables
- add validation and tests

### Stage 3: Native MLX Training Primitives

- expose checkpointing and value-and-grad compilation paths
- add a backend-facing training API or extension

### Stage 4: MLX-Specific Large-Training Strategy

- prototype checkpointed blockwise training
- document recommended routing vs MPSGraph

### Stage 5: Capability and Risk Heuristics

- tighten capability contracts
- add tensor-shape early-risk detection

### Stage 6: Qualification Matrix

- publish safe/unsafe zones by workload class

## 12. Concrete File-Level Plan

Likely touch points:

- `mlx/backend.go`
- `mlx/builder.go`
- `mlx/executable.go`
- `mlx/function.go`
- `mlx/internal/bridge/bridge.go`
- `mlx/capabilities.go`
- `mlx/memory_smoke_test.go`
- possibly a new `mlx/training_extensions.go`

Recommended additions:

- `mlx/execution_mode.go`
- `mlx/training_api.go`
- `mlx/checkpoint_test.go`
- `mlx/donation_test.go`
- `mlx/large_graph_qualification.md`

## 13. Risk Register

### High Risk

- assuming the remaining 14.3 GB limit is “just another leak”
- continuing to route MLX through the MPSGraph-style segmented VJP strategy
- silent Go-callback fallback in training graphs
- backend claiming support that is only partial in practice

### Medium Risk

- donation semantics interacting poorly with shared buffers
- checkpoint integration requiring upstream GoMLX API cooperation
- shape-dependent `DotGeneral` behavior producing non-obvious memory cliffs

### Low Risk

- bridge helper cleanup
- memory logging and guardrail work
- capability metadata refinements

## 14. Suggested Milestones

### Milestone A: Safe MLX Backend

- configurable memory policy
- wired-limit support
- donation implemented
- fallback path visibility added

### Milestone B: Honest Capability Model

- backend differentiates supported vs train-safe vs ctape-safe
- strict training mode available

### Milestone C: MLX-Native Checkpointed Training

- checkpointed autograd closure integrated
- at least one training path uses MLX-native differentiation intentionally

### Milestone D: Large-Training Qualification

- documented safe zones
- explicit recommendation boundaries vs MPSGraph

## Recommendation

Treat the next phase as a **backend strategy correction**, not just bug cleanup.

The MLX backend is already much healthier than it was during the crash/leak phase. The remaining gap is that it still lacks the memory-management strategy needed for truly large training jobs.

The most valuable next steps are:

1. safety and visibility,
2. donation,
3. checkpointed MLX-native autograd integration,
4. and explicit separation of MLX strategy from MPSGraph strategy.

If those land, MLX can become a trustworthy backend for more than just small/medium training. Without them, it will remain operationally useful but structurally mismatched to your largest fused chunker workloads.
