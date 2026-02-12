# MPSGraph Backend for GoMLX — Progress

## Overview

An Apple Metal GPU backend for [GoMLX](https://github.com/gomlx/gomlx) using Apple's MPSGraph framework. Lives inside the [go-coreml](https://github.com/gomlx/go-coreml) module at `mpsgraph/gomlx/`.

**Goal**: GPU-accelerated training on Apple Silicon, targeting significant speedup over CPU-only execution.

**Status**: Feature-complete for training standard models (dense, CNN, transformer). All 59 tests passing.

---

## Current Status: Phase 3 Complete

### Benchmarks (Apple M4 Pro)

| Benchmark | Time | Notes |
|-----------|------|-------|
| MatMul 512×256×512 | 0.36 ms | **4.9× faster** than SimpleGo CPU (1.75 ms) |
| MatMul 1024×768×1024 | 1.22 ms | ~1.3 TFLOPS |
| MatMul 2048×768×2048 | 3.66 ms | ~1.7 TFLOPS |
| Transformer training step (B8, S128, D64) | 2.88 ms | LayerNorm → Dense → GELU → Dense + Adam |

### Supported Operations (~115 ops)

#### Inputs (2)
- Parameter, Constant

#### Unary Math (22)
- Abs, Neg, Sqrt, Rsqrt, Exp, Expm1, Log, Log1p, Sin, Cos, Tanh, Logistic (Sigmoid), Erf, Floor, Ceil, Round, Sign, LogicalNot, BitwiseNot, IsFinite, IsNaN, Identity

#### Binary Math (17)
- Add, Sub, Mul, Div, Rem, Pow, Max, Min, Atan2
- LogicalAnd, LogicalOr, LogicalXor
- BitwiseAnd, BitwiseOr, BitwiseXor
- ShiftLeft, ShiftRightArithmetic, ShiftRightLogical

#### Comparison (12)
- Equal, NotEqual, GreaterThan, GreaterOrEqual, LessThan, LessOrEqual
- EqualTotalOrder, NotEqualTotalOrder, GreaterThanTotalOrder, GreaterOrEqualTotalOrder, LessThanTotalOrder, LessOrEqualTotalOrder

#### Shape Operations (11)
- Reshape, Transpose, ConvertDType, BroadcastInDim, Where, Clamp, Slice, Concatenate, Reverse, Iota, Pad

#### Matrix Operations (2)
- Dot, DotGeneral (with full axis decomposition following SimpleGo's pattern)

#### Reductions (6)
- ReduceSum, ReduceMax, ReduceMin, ReduceProduct
- ReduceLogicalAnd, ReduceLogicalOr

#### Index Operations (5)
- Gather, ScatterSum, ScatterMax, ScatterMin
- ArgMinMax

#### Dynamic Indexing (2)
- DynamicSlice, DynamicUpdateSlice

#### Convolution (1)
- ConvGeneral (with input dilation support for gradient computation)

#### Pooling (2)
- ReduceWindow (MaxPool, SumPool, AvgPool — both NCHW and NHWC)
- SelectAndScatterMax (MaxPool gradient — both NCHW and NHWC)

#### Normalization (3)
- BatchNormForInference
- BatchNormForTraining (decomposed: mean → variance → normalize → scale + offset)
- BatchNormGradient (decomposed: standard batch norm gradient formulas)

#### Random (1)
- RNGBitGenerator (Philox-based)

#### Control Flow (4)
- While (CPU-orchestrated loop with compiled cond/body closures)
- If (CPU-orchestrated branch selection with compiled true/false closures)
- Sort (CPU-orchestrated sort with compiled comparator closure)
- Call (invokes a named sub-function)

#### Fused Operations (6)
- FusedSoftmax (native MPSGraph softmax)
- FusedLayerNorm (decomposed: mean → variance → normalize → scale + shift)
- FusedGelu (exact and approximate)
- FusedDense (decomposed: DotGeneral + bias + activation; supports None/Gelu/Relu/Silu/Tanh)
- FusedAttentionQKVProjection (decomposed: DotGeneral + slice + per-head bias)
- FusedScaledDotProductAttention (decomposed: reshape + batched matmul + mask + softmax + matmul; supports BHSD/BSHD layouts, GQA, causal masking, boolean/additive masks)

### Supported Data Types (11)
- Float32, Float16, BFloat16
- Bool
- Int8, Int16, Int32, Int64
- Uint8, Uint16, Uint32

### Training Integration
- End-to-end training validated with:
  - **Linear regression** (SGD optimizer)
  - **Dense network** (Adam optimizer)
  - **MNIST linear model**
  - **CNN model** (Conv2D + ReLU + MaxPool + Flatten + Dense)
  - **Transformer-like model** (LayerNorm + Dense + GELU + mean pooling + Dense)
- Automatic differentiation works through all supported ops (GoMLX's VJP system)
- Gradient tests pass for simple ops, dense layers, maxpool backprop, and layer normalization

### Test Summary (59 tests)

| Category | Tests | What's Covered |
|----------|-------|---------------|
| Backend basics | 3 | Creation, buffer ops, shared buffers |
| Core ops | 17 | Add, unary, binary, reshape, constant, reduce, dot, gather, slice, concat, iota, broadcast, pad, argminmax, dynamic slice/update, comparison, where |
| Fused ops (Phase 1-2) | 3 | FusedSoftmax, FusedLayerNorm, FusedGelu |
| Advanced ops | 5 | Scatter, reverse, reduce product, clamp, logical reductions |
| GoMLX graph API | 8 | Add, math chain, matmul, reduce+broadcast, gather+slice, where+compare, convolution, pooling |
| Training | 7 | Context exec, linear regression, Adam, gradients (simple + dense), MNIST, CNN |
| Transformer | 1 | Full training loop with LayerNorm + Dense + GELU + Adam |
| Integer/dtype | 2 | Int64 ops, BFloat16 |
| Fused ops (Phase 3) | 3 | FusedDense (5 sub-tests), FusedAttentionQKVProjection (2 sub-tests), FusedScaledDotProductAttention (4 sub-tests) |
| BatchNorm | 1 | BatchNormForTraining (2 sub-tests) |
| Control flow | 4 | While (2 sub-tests), If (3 sub-tests), Sort (2 sub-tests), Call (1 sub-test) |
| Benchmarks | 3 | MatMul sizes, unary chain |

---

## Architecture

### Layer 1: Objective-C++ Bridge (`internal/bridge/`)
- `bridge.h` — C-compatible MPSGraph function declarations
- `bridge.m` — Objective-C++ implementation wrapping MPSGraph APIs
- `bridge.go` — Go CGo bindings (`darwin && cgo`)
- `bridge_nocgo.go` — Stubs when CGO is disabled

### Layer 2: GoMLX Backend (`gomlx/`)
- `backend.go` — `backends.Backend` implementation (device management, registration)
- `builder.go` — `backends.Builder` implementation (graph compilation, control flow graph segmentation)
- `function.go` — `backends.Function` + `backends.FusedOps` (~115 op implementations, closure/capture support)
- `executable.go` — `backends.Executable` (compiled graph execution) + `ExecutableWithCF` (control flow orchestration)
- `buffer.go` — Buffer management using MTLBuffer with shared/unified memory
- `capabilities.go` — Declared supported ops, dtypes, and `Functions: true`
- `dotgeneral.go` — DotGeneral decomposition (transpose + reshape + matmul)
- `gather.go` — XLA Gather/Scatter semantics translation to MPSGraph
- `register_darwin.go` — `init()` registers backend as `"mpsgraph"`

### Key Design Decisions
- **Unified memory**: `HasSharedBuffers() = true` — MTLBuffer with `storageModeShared` for zero-copy CPU↔GPU
- **No gradient implementation needed**: GoMLX's VJP system decomposes gradients at graph-build time; backend just executes forward ops
- **NCHW/NHWC auto-detection**: Pooling ops detect spatial axes from window dimensions and auto-transpose
- **Decomposed fused ops**: Fused ops (Dense, Attention, BatchNorm) are decomposed into primitive ops at graph-build time using existing MPSGraph operations; GoMLX's `InternalFusedOpCaller` provides fallback decomposition if backend returns `ErrNotImplemented`
- **CPU-orchestrated control flow**: While/If/Sort/Call are implemented by splitting the computation graph at control flow boundaries — pre-CF graph compiles to one MPSGraph executable, each closure compiles to its own MPSGraph executable (with its own `bridge.Context`), and the CPU orchestrates execution by reading intermediate results and making branch/loop decisions. Named functions (Call) compile from the builder's shared context; closures compile from their own independent context. Value capture is handled transparently: when a closure references a parent scope value, a placeholder feed tensor is created in the closure's context and the parent value is passed at execution time.

---

## Phase History

### Phase 1 — Core Operations
- Basic infrastructure (backend, builder, function, executable, buffer)
- Unary/binary math, comparisons, shape ops, reductions
- DotGeneral, Gather/Scatter, BatchNormForInference

### Phase 2 — Training Support
- Convolution with input dilation (gradient support)
- Pooling (ReduceWindow, SelectAndScatterMax) with NCHW/NHWC auto-detection
- RNG, DynamicSlice/DynamicUpdateSlice
- FusedSoftmax, FusedLayerNorm, FusedGelu
- TotalOrder comparisons, logical reductions
- End-to-end training validation (84 tests passing)
- Code review fixes: ReduceWindow sum, ShiftRightLogical, FusedGelu exact mode, dilation validation, ScatterMode consistency, Builder.Finalize, dtypeToBridgeDType panic, gather transpose fix

### Phase 3 — Fused Ops, BatchNorm, Control Flow, BFloat16
- **BFloat16** data type support (11 dtypes total)
- **FusedDense** — matmul + optional bias + activation (None/Gelu/Relu/Silu/Tanh)
- **FusedAttentionQKVProjection** — single matmul → slice into Q/K/V + optional per-head biases
- **FusedScaledDotProductAttention** — full attention: head reshape + batched Q·K^T + scale + mask + softmax + V multiply; supports BHSD and BSHD axis layouts, grouped-query attention (GQA), causal masking, boolean and additive masks
- **BatchNormForTraining** — decomposed: reduce mean → reduce variance → normalize → scale + offset; returns (normalized, mean, variance)
- **BatchNormGradient** — decomposed: standard batch norm gradient formulas for (gradOperand, gradScale, gradOffset)
- **Control flow** — While, If, Sort, Call with CPU-orchestrated execution:
  - Each closure gets its own `bridge.Context` (independent MPSGraph)
  - Value capture: `resolveNode` intercepts parent scope references, creates placeholder feeds in closure context
  - Graph segmentation: pre-CF graph → compile → execute → read CF inputs → orchestrate CF → assemble outputs
  - While: CPU loop evaluating compiled cond closure, executing compiled body closure
  - If: CPU reads scalar bool predicate, selects and executes compiled branch closure
  - Sort: CPU `sort.Slice`/`sort.SliceStable` with compiled comparator closure for element-pair comparison
  - Call: executes compiled named function with inputs + captured values
- **59 tests passing** (39 original + 20 new Phase 3 tests)

---

## Not Yet Implemented

### Operations
- FFT
- Complex number operations (Complex, Conj, Real, Imag)
- Bitcast, BitCount, Clz
- ReduceBitwiseAnd/Or/Xor, ReduceLogicalXor
- SelectAndScatterMin
- AllReduce (distributed computing)

### Data Types
- Float64 (limited MPSGraph support)
- Uint64
- Complex64, Complex128

---

## File Listing

```
mpsgraph/gomlx/
├── backend.go              # Backend struct, New(), Register(), Finalize()
├── buffer.go               # MTLBuffer-backed shared buffers
├── builder.go              # Builder struct, Compile(), compileWithControlFlow()
├── capabilities.go         # Supported ops, dtypes, Functions: true
├── dotgeneral.go           # DotGeneral axis decomposition
├── executable.go           # Executable + ExecutableWithCF (control flow orchestration)
├── function.go             # All ~115 op implementations + closure/capture support
├── gather.go               # XLA Gather/Scatter translation
├── go.mod / go.sum         # Separate Go module
├── mpsgraph_test.go        # 59 tests + 3 benchmarks
├── nocgo_darwin.go         # CGO-disabled stub
├── register_darwin.go      # init() → backends.Register("mpsgraph", New)
├── stub_other.go           # Non-Darwin stub
├── Progress.md             # This file
└── internal/bridge/
    ├── bridge.h            # C API declarations
    ├── bridge.m            # Objective-C++ MPSGraph implementation
    ├── bridge.go           # Go CGo bindings
    └── bridge_nocgo.go     # CGO-disabled stubs
```

---

## Next Steps

1. **Real model validation** — Run DeBERTa or similar GoMLX transformer model end-to-end on MPSGraph
2. **Native MPSGraph SDPA** — Use `MPSGraphScaledDotProductAttention` (macOS 15+) for flash-attention-style memory savings
3. **Native MPSGraph fused operations** — Replace decomposed fused ops with native equivalents for performance
4. **Float64 support** — Some scientific computing use cases
5. **Performance profiling** — Identify bottlenecks in real training workloads (memory, kernel launch overhead, CPU↔GPU transfers)
