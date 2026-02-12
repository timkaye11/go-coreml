# MPSGraph Backend for GoMLX — Progress

## Overview

An Apple Metal GPU backend for [GoMLX](https://github.com/gomlx/gomlx) using Apple's MPSGraph framework. Lives inside the [go-coreml](https://github.com/gomlx/go-coreml) module at `mpsgraph/gomlx/`.

**Goal**: GPU-accelerated training on Apple Silicon, targeting significant speedup over CPU-only execution.

---

## Current Status: Phase 2 Complete — 84 Tests Passing

### Benchmarks (Apple M4 Pro)

| Benchmark | Time | Notes |
|-----------|------|-------|
| MatMul 512×256×512 | 0.36 ms | **4.9× faster** than SimpleGo CPU (1.75 ms) |
| MatMul 1024×768×1024 | 1.22 ms | ~1.3 TFLOPS |
| MatMul 2048×768×2048 | 3.66 ms | ~1.7 TFLOPS |
| Transformer training step (B8, S128, D64) | 2.88 ms | LayerNorm → Dense → GELU → Dense + Adam |

### Supported Operations (~100 ops)

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

#### Normalization (1)
- BatchNormForInference

#### Random (1)
- RNGBitGenerator (Philox-based)

#### Fused Operations (3)
- FusedSoftmax (native MPSGraph softmax)
- FusedLayerNorm (decomposed: mean → variance → normalize → scale + shift)
- FusedGelu (approximation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`)

### Supported Data Types (10)
- Float32, Float16
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

---

## Architecture

### Layer 1: Objective-C++ Bridge (`internal/bridge/`)
- `bridge.h` — C-compatible MPSGraph function declarations
- `bridge.m` — Objective-C++ implementation wrapping MPSGraph APIs
- `bridge.go` — Go CGo bindings (`darwin && cgo`)
- `bridge_nocgo.go` — Stubs when CGO is disabled

### Layer 2: GoMLX Backend (`gomlx/`)
- `backend.go` — `backends.Backend` implementation (device management, registration)
- `builder.go` — `backends.Builder` implementation (graph compilation)
- `function.go` — `backends.Function` + `backends.FusedOps` (all op implementations)
- `executable.go` — `backends.Executable` (compiled graph execution)
- `buffer.go` — Buffer management using MTLBuffer with shared/unified memory
- `capabilities.go` — Declared supported ops and dtypes
- `dotgeneral.go` — DotGeneral decomposition (transpose + reshape + matmul)
- `gather.go` — XLA Gather/Scatter semantics translation to MPSGraph
- `register_darwin.go` — `init()` registers backend as `"mpsgraph"`

### Key Design Decisions
- **Unified memory**: `HasSharedBuffers() = true` — MTLBuffer with `storageModeShared` for zero-copy CPU↔GPU
- **No gradient implementation needed**: GoMLX's VJP system decomposes gradients at graph-build time; backend just executes forward ops
- **NCHW/NHWC auto-detection**: Pooling ops detect spatial axes from window dimensions and auto-transpose

---

## Not Yet Implemented

### Operations
- FFT
- Complex number operations (Complex, Conj, Real, Imag)
- Control flow (While, If, Sort, Call) — requires `Functions: true`
- Bitcast, BitCount, Clz
- ReduceBitwiseAnd/Or/Xor, ReduceLogicalXor
- SelectAndScatterMin
- BatchNormForTraining, BatchNormGradient (has manual fallback via decomposed graph)
- AllReduce (distributed computing)
- FusedDense, FusedScaledDotProductAttention, FusedAttentionQKVProjection

### Data Types
- Float64 (limited MPSGraph support)
- BFloat16 (requires M4+ / macOS 15+)
- Uint64
- Complex64, Complex128

---

## File Listing

```
mpsgraph/gomlx/
├── backend.go              # Backend struct, New(), Register(), Finalize()
├── buffer.go               # MTLBuffer-backed shared buffers
├── builder.go              # Builder struct, Compile()
├── capabilities.go         # Supported ops and dtypes declaration
├── dotgeneral.go           # DotGeneral axis decomposition
├── executable.go           # Executable with Execute()
├── function.go             # All ~100 op implementations
├── gather.go               # XLA Gather/Scatter translation
├── go.mod / go.sum         # Separate Go module
├── mpsgraph_test.go        # 84 tests + benchmarks
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

1. **FusedScaledDotProductAttention** — Key performance win for transformer models
2. **BFloat16 support** — Performance boost on M4+ chips
3. **Real model benchmarking** — Run DeBERTa or similar transformer end-to-end
4. **Control flow ops** (While, If) — Needed for dynamic models
5. **Float64 support** — Some scientific computing use cases
