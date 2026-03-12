# MLX Backend Implementation Plan

## Overview

This document describes the plan to add an Apple MLX backend to `go-coreml`, alongside the existing MPSGraph backend. The MLX backend targets **training and finetuning** workloads on Apple Silicon, leveraging MLX's automatic differentiation, custom Metal shaders, unified memory, and JIT compilation.

The backend bridges Go → CGo → [mlx-c](https://github.com/ml-explore/mlx-c) → MLX (C++).

---

## Why MLX (Not More MPSGraph)

| Concern | MPSGraph (current) | MLX (proposed) |
|---------|-------------------|----------------|
| Autograd | None — would need manual gradient ops for every operation | Built-in `mlx_value_and_grad`, `mlx_vjp`, `mlx_jvp` |
| Memory | Go heap → copy → MTLBuffer → copy back per execution | Unified memory — zero-copy between CPU/GPU |
| Metal approach | Apple's MPSGraph framework (high-level abstraction) | Custom hand-tuned Metal shaders |
| JIT compilation | Static graph compile | `mlx_compile()` fuses operations, eliminates intermediates |
| Quantization | None | Built-in `mlx_quantize`/`mlx_dequantize`/`mlx_quantized_matmul` |
| Operations | ~46 | 400+ via mlx-c |
| Training proven | No | LLaMA LoRA finetuning in MLX examples |
| RoPE | Manual decomposition | Native `mlx_fast_rope()` Metal kernel |
| Attention | Fused via MPSGraph ops | `mlx_fast_scaled_dot_product_attention()` with custom Metal kernels |

Adding training to MPSGraph would require implementing gradient operations for every op in Objective-C++. MLX provides all of this through a stable C API.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    GoMLX Application                      │
│               (gopeft training loop, model def)           │
├──────────────────────────────────────────────────────────┤
│               backends.Backend interface                   │
│   Backend, Builder, Function, Executable, Buffer          │
│   StandardOps (150+ op types), FusedOps, CollectiveOps   │
├──────────────────────────────────────────────────────────┤
│              mlx/gomlx/  (NEW — this plan)                │
│   backend.go  builder.go  function.go  buffer.go         │
│   executable.go  capabilities.go  register_darwin.go     │
├──────────────────────────────────────────────────────────┤
│              mlx/gomlx/internal/bridge/                   │
│   bridge.go  (CGo → mlx-c)                               │
│   bridge_nocgo.go  (stubs for non-CGo builds)            │
├──────────────────────────────────────────────────────────┤
│              mlx-c  (C API — linked as static library)    │
│   mlx_array, mlx_ops, mlx_transforms, mlx_compile        │
│   400+ ops, autograd, compilation, closures               │
├──────────────────────────────────────────────────────────┤
│              MLX  (C++ core)                              │
│   Custom Metal shaders, unified memory,                   │
│   lazy evaluation, JIT, autograd engine                   │
├──────────────────────────────────────────────────────────┤
│              Apple Metal / Apple Silicon GPU               │
└──────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Mirror the MPSGraph backend structure** — same file layout, same patterns, easier to maintain
2. **Unified memory everywhere** — no Go-side buffer copies; `mlx_array` IS the buffer
3. **Lazy graph → compiled closure** — GoMLX's build-then-execute maps to MLX's lazy eval + `mlx_compile()`
4. **mlx-c is the only C dependency** — no direct Metal/Obj-C code (MLX handles all GPU interaction)
5. **Training via mlx-c transforms** — `mlx_value_and_grad()` for autograd, exposed as backend extension

---

## Directory Structure

```
go-coreml/
├── mlx/                                    # NEW top-level module
│   └── gomlx/
│       ├── go.mod                          # module github.com/gomlx/go-coreml/mlx/gomlx
│       ├── go.sum
│       │
│       ├── backend.go                      # Backend interface implementation
│       ├── builder.go                      # Builder — creates Functions, compiles to Executable
│       ├── buffer.go                       # Buffer backed by mlx_array (unified memory)
│       ├── capabilities.go                 # Supported ops & dtypes declaration
│       ├── executable.go                   # Executable — runs compiled MLX closures
│       ├── function.go                     # Function — all StandardOps implementations
│       ├── function_fused.go               # FusedOps implementations (attention, layernorm, etc.)
│       ├── dotgeneral.go                   # DotGeneral decomposition (port from mpsgraph)
│       ├── gather.go                       # Gather decomposition (port from mpsgraph)
│       ├── register_darwin.go              # Backend auto-registration on macOS
│       ├── nocgo_darwin.go                 # Stub when CGo disabled
│       ├── stub_other.go                   # Stub on non-macOS platforms
│       ├── mlx_test.go                     # Comprehensive test suite
│       │
│       └── internal/
│           └── bridge/
│               ├── bridge.go              # CGo bindings to mlx-c (MAIN FILE)
│               ├── bridge_nocgo.go        # Stubs for non-CGo builds
│               └── deps/                  # Build artifacts (see Build System)
│                   ├── include/           # mlx-c headers (mlx/c/*.h)
│                   └── lib/               # libmlx.a, libmlxc.a
│
├── mpsgraph/                               # EXISTING (unchanged)
│   └── gomlx/ ...
│
└── scripts/
    └── build_mlx.sh                        # Script to build mlx-c from source
```

---

## Build System

### mlx-c Build Script (`scripts/build_mlx.sh`)

```bash
#!/bin/bash
# Builds mlx + mlx-c as static libraries for CGo linking.
#
# Prerequisites: Xcode, CMake, C++20 compiler
# Output: mlx/gomlx/internal/bridge/deps/{include,lib}

set -euo pipefail

MLX_C_VERSION="v0.5.0"      # Pin to specific release
BUILD_DIR="/tmp/mlx-c-build"
DEST_DIR="$(dirname "$0")/../mlx/gomlx/internal/bridge/deps"

# Clone and build
git clone --depth 1 --branch "$MLX_C_VERSION" https://github.com/ml-explore/mlx-c.git "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_C_BUILD_SHARED=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="$DEST_DIR"

cmake --build build --parallel $(sysctl -n hw.ncpu)
cmake --install build

echo "MLX-C built and installed to $DEST_DIR"
```

### CGo Linking Directives (`bridge.go`)

```go
//go:build darwin && cgo

/*
#cgo darwin CFLAGS: -I${SRCDIR}/deps/include
#cgo darwin LDFLAGS: -L${SRCDIR}/deps/lib -lmlxc -lmlx -lc++ -framework Metal -framework Foundation -framework Accelerate
#include "mlx/c/mlx.h"
*/
import "C"
```

### go.mod

```
module github.com/gomlx/go-coreml/mlx/gomlx

go 1.25

require (
    github.com/gomlx/gomlx v0.26.1-0.20260223064152-358aaf0bc270
    github.com/pkg/errors v0.9.1
)
```

---

## Phase-by-Phase Implementation

### Phase 1: Bridge Foundation & Buffer Layer

**Goal**: CGo compiles, arrays can be created/destroyed, data flows between Go and MLX.

#### bridge.go — Core Types

```go
package bridge

/*
#cgo darwin CFLAGS: -I${SRCDIR}/deps/include
#cgo darwin LDFLAGS: -L${SRCDIR}/deps/lib -lmlxc -lmlx -lc++ -framework Metal -framework Foundation -framework Accelerate
#include "mlx/c/mlx.h"
#include <stdlib.h>
*/
import "C"
import (
    "unsafe"
    "runtime"
    "github.com/pkg/errors"
)

// Array wraps an mlx_array handle with Go-side reference management.
type Array struct {
    handle C.mlx_array
}

// Stream wraps mlx_stream for dispatching operations to CPU or GPU.
type Stream struct {
    handle C.mlx_stream
}

// Closure wraps mlx_closure for compiled functions.
type Closure struct {
    handle C.mlx_closure
}
```

Key functions to implement:

| Function | mlx-c call | Purpose |
|----------|-----------|---------|
| `NewArrayFromData(data unsafe.Pointer, shape []int, dtype DType) *Array` | `mlx_array_new_data` | Create array from Go slice data |
| `ArrayFree(a *Array)` | `mlx_array_free` | Release array |
| `ArrayShape(a *Array) []int` | `mlx_array_shape` | Query dimensions |
| `ArrayDType(a *Array) DType` | `mlx_array_dtype` | Query element type |
| `ArrayData(a *Array) unsafe.Pointer` | `mlx_array_data_*` | Get pointer to unified memory |
| `ArrayNBytes(a *Array) int` | `mlx_array_nbytes` | Byte count |
| `Eval(arrays ...*Array)` | `mlx_eval` | Force lazy computation |
| `DefaultGPUStream() *Stream` | `mlx_default_gpu_stream_new` | Get GPU stream |
| `DefaultCPUStream() *Stream` | `mlx_default_cpu_stream_new` | Get CPU stream |
| `MetalIsAvailable() bool` | `mlx_metal_is_available` | Check Metal support |

**DType mapping**:

| GoMLX dtype | mlx-c dtype constant |
|-------------|---------------------|
| dtypes.Bool | `MLX_BOOL` |
| dtypes.Float16 | `MLX_FLOAT16` |
| dtypes.BFloat16 | `MLX_BFLOAT16` |
| dtypes.Float32 | `MLX_FLOAT32` |
| dtypes.Float64 | `MLX_FLOAT64` (note: MLX supports this natively) |
| dtypes.Int8 | `MLX_INT8` |
| dtypes.Int16 | `MLX_INT16` |
| dtypes.Int32 | `MLX_INT32` |
| dtypes.Int64 | `MLX_INT64` |
| dtypes.Uint8 | `MLX_UINT8` |
| dtypes.Uint16 | `MLX_UINT16` |
| dtypes.Uint32 | `MLX_UINT32` |
| dtypes.Uint64 | `MLX_UINT64` |
| dtypes.Complex64 | `MLX_COMPLEX64` |

#### buffer.go — Unified Memory Buffers

```go
// mlxBuffer wraps an MLX array as a GoMLX buffer.
// Unlike the MPSGraph backend (which copies to/from Go heap),
// mlxBuffer uses MLX's unified memory — the array IS the buffer.
type mlxBuffer struct {
    array *bridge.Array
    shape shapes.Shape
}
```

Key design difference from MPSGraph:

- **MPSGraph**: `gpuBuffer.flat` is a Go slice. Data copied Go→Metal on execute, Metal→Go on read-back.
- **MLX**: `mlxBuffer.array` is an `mlx_array` in unified memory. `BufferData()` returns a Go slice pointing directly into unified memory. **No copies**.

**HasSharedBuffers() → true**: Because MLX arrays live in shared memory accessible from both CPU (Go) and GPU.

**BufferFromFlatData**: Creates an `mlx_array` from Go data. This is the ONE copy (Go heap → unified memory). After that, all operations are zero-copy.

**BufferToFlatData**: Calls `mlx_eval()` to materialize (if lazy), then copies from unified memory to caller's Go slice.

**BufferData** (shared buffer access): Returns a Go slice whose backing memory IS the mlx_array's unified memory. Requires `mlx_eval()` first to materialize.

#### backend.go — Backend Shell

```go
const BackendName = "mlx"

type Backend struct {
    gpuStream  *bridge.Stream
    cpuStream  *bridge.Stream
    deviceName string
    mu         sync.RWMutex
    isFinalized bool
}

var _ backends.Backend = &Backend{}
```

Implement all `backends.Backend` methods:
- `New(config string) (backends.Backend, error)` — check `MetalIsAvailable()`, create streams
- `Name() → "mlx"`
- `Builder(name string) → Builder`
- `Capabilities() → backendCapabilities` (see Phase 2)
- All `Buffer*` methods delegating to mlxBuffer
- `Finalize()` — free streams, clear MLX cache via `mlx_clear_cache()`

#### register_darwin.go

```go
//go:build darwin && cgo

package mlx

import "github.com/gomlx/gomlx/backends"

func init() {
    backends.Register(BackendName, New)
}
```

#### Validation

- [ ] `go build ./mlx/gomlx/...` compiles on macOS with CGo
- [ ] Create array from Go float32 slice, read back, verify values match
- [ ] `MetalIsAvailable()` returns true on Apple Silicon
- [ ] Array creation + free cycle doesn't leak (test with ASAN or `mlx_get_active_memory()`)

---

### Phase 2: Element-wise & Arithmetic Operations

**Goal**: All unary, binary, comparison, and logical ops working.

#### capabilities.go

Declare all supported operations and data types. MLX supports significantly more than MPSGraph:

```go
var backendCapabilities = backends.Capabilities{
    Functions: true, // MLX supports closures via mlx_closure
    DTypes: map[dtypes.DType]bool{
        dtypes.Bool:      true,
        dtypes.Float16:   true,
        dtypes.BFloat16:  true,
        dtypes.Float32:   true,
        dtypes.Float64:   true, // Native support (unlike MPSGraph which downcasts)
        dtypes.Int8:      true,
        dtypes.Int16:     true,
        dtypes.Int32:     true,
        dtypes.Int64:     true,
        dtypes.Uint8:     true,
        dtypes.Uint16:    true,
        dtypes.Uint32:    true,
        dtypes.Uint64:    true,
        dtypes.Complex64: true, // MLX supports complex (MPSGraph doesn't)
    },
    Operations: map[backends.OpType]bool{
        // All 150+ OpTypes — see full list in implementation
    },
}
```

#### bridge.go — Arithmetic Operations

Each op follows the mlx-c pattern:

```go
func Add(a, b *Array, stream *Stream) (*Array, error) {
    result := &Array{}
    rc := C.mlx_add(&result.handle, a.handle, b.handle, stream.handle)
    if rc != 0 {
        return nil, errors.New("mlx_add failed")
    }
    return result, nil
}
```

**Operations to bridge (Phase 2)**:

| Category | bridge.go functions | mlx-c calls |
|----------|-------------------|-------------|
| Unary math | `Abs`, `Neg`, `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Log`, `Log1p`, `Erf`, `Sigmoid`, `Floor`, `Ceil`, `Round`, `Sign` | `mlx_abs`, `mlx_negative`, `mlx_sqrt`, `mlx_rsqrt`, `mlx_exp`, `mlx_expm1`, `mlx_log`, `mlx_log1p`, `mlx_erf`, `mlx_sigmoid`, `mlx_floor`, `mlx_ceil`, `mlx_round`, `mlx_sign` |
| Trig | `Sin`, `Cos`, `Tanh` | `mlx_sin`, `mlx_cos`, `mlx_tanh` |
| Binary math | `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Pow`, `Maximum`, `Minimum`, `Atan2` | `mlx_add`, `mlx_subtract`, `mlx_multiply`, `mlx_divide`, `mlx_remainder`, `mlx_power`, `mlx_maximum`, `mlx_minimum`, `mlx_arctan2` |
| Comparison | `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual` | `mlx_equal`, `mlx_not_equal`, `mlx_less`, `mlx_less_equal`, `mlx_greater`, `mlx_greater_equal` |
| Logical | `LogicalAnd`, `LogicalOr`, `LogicalNot` | `mlx_logical_and`, `mlx_logical_or`, `mlx_logical_not` |
| Bitwise | `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, `ShiftLeft`, `ShiftRight` | Direct mlx-c equivalents |
| Special | `IsNaN`, `IsFinite`, `Identity` | `mlx_isnan`, `mlx_isinf` (invert), identity via copy |

#### function.go — StandardOps Implementation

Map each GoMLX `StandardOps` method to the bridge:

```go
func (f *Function) Add(lhs, rhs backends.Value) (backends.Value, error) {
    a, b := f.resolve(lhs), f.resolve(rhs)
    result, err := bridge.Add(a.array, b.array, f.stream())
    if err != nil {
        return nil, err
    }
    return f.newNode(result, /* infer shape */), nil
}
```

**Graph node tracking**: Like MPSGraph, each operation returns a `*graphNode` that wraps the result `bridge.Array` and its shape. Since MLX is lazy, no computation happens yet — just graph construction.

#### Validation

- [ ] All unary ops: create input, apply op, `Eval()`, verify against Go reference
- [ ] Binary ops with broadcasting
- [ ] Comparison ops return bool arrays
- [ ] Port relevant tests from `mpsgraph_test.go`

---

### Phase 3: Shape, Array, and Reduction Operations

#### bridge.go — Shape Operations

| Function | mlx-c call | Notes |
|----------|-----------|-------|
| `Reshape(a, shape)` | `mlx_reshape` | |
| `Transpose(a, axes)` | `mlx_transpose` | |
| `BroadcastTo(a, shape)` | `mlx_broadcast_to` | |
| `Concatenate(arrays, axis)` | `mlx_concatenate` | Uses `mlx_vector_array` |
| `Slice(a, starts, stops, strides)` | `mlx_slice` | Static slicing |
| `Pad(a, axes, widths, value)` | `mlx_pad` | MLX supports more padding modes |
| `Reverse(a, axes)` | `mlx_flip` | MLX calls it "flip" |
| `Where(cond, a, b)` | `mlx_where` | Element-wise select |
| `Clamp(a, min, max)` | `mlx_clip` | MLX calls it "clip" |
| `ConvertDType(a, dtype)` | `mlx_astype` | |
| `Take(a, indices, axis)` | `mlx_take` | Richer than MPSGraph gather |
| `Scatter(a, indices, updates, ...)` | `mlx_scatter` + variants | |

#### bridge.go — Reduction Operations

| Function | mlx-c call |
|----------|-----------|
| `ReduceSum(a, axes, keepdims)` | `mlx_sum_axes` |
| `ReduceMax(a, axes, keepdims)` | `mlx_max_axes` |
| `ReduceMin(a, axes, keepdims)` | `mlx_min_axes` |
| `ReduceProduct(a, axes, keepdims)` | `mlx_prod_axes` |
| `ArgMin(a, axis, keepdims)` | `mlx_argmin` |
| `ArgMax(a, axis, keepdims)` | `mlx_argmax` |

#### bridge.go — Special Operations

| Function | mlx-c call | Notes |
|----------|-----------|-------|
| `Iota(shape, axis)` | `mlx_arange` + reshape | MLX uses arange, reshape to match GoMLX iota |
| `DynamicSlice` | `mlx_slice` with computed indices | |
| `DynamicUpdateSlice` | `mlx_slice_update` | |

#### gather.go — Gather/Scatter

MLX has `mlx_take`, `mlx_take_along_axis`, `mlx_gather` which are **much richer** than MPSGraph's limited gather. Strategy:

1. Simple embedding lookups → `mlx_take(operand, indices, axis)`
2. Gather along axis → `mlx_take_along_axis(operand, indices, axis)`
3. Complex XLA-style Gather → decompose using same logic as `mpsgraph/gomlx/gather.go` but with MLX primitives

#### Validation

- [ ] Reshape, transpose, concat with various ranks
- [ ] Reductions along different axes, with keepdims
- [ ] Gather/scatter matching MPSGraph test cases
- [ ] Iota generation

---

### Phase 4: Linear Algebra

#### bridge.go — Matrix Operations

| Function | mlx-c call | Notes |
|----------|-----------|-------|
| `MatMul(a, b)` | `mlx_matmul` | Batched automatically |
| `Einsum(equation, inputs...)` | `mlx_einsum` | **Full arbitrary einsum** (MPSGraph only supports limited patterns) |

#### dotgeneral.go

Port from `mpsgraph/gomlx/dotgeneral.go`. Same decomposition strategy:

1. Detect simple matmul → direct `bridge.MatMul()`
2. General case: classify axes → transpose → reshape to rank-3 → batched matmul → reshape back

The logic is identical; only the bridge calls change (MPSGraph → MLX).

#### Validation

- [ ] MatMul: various ranks, batched
- [ ] DotGeneral: contracting axes, batch axes, cross axes
- [ ] Einsum: test patterns that fail on MPSGraph (MLX supports all patterns)

---

### Phase 5: Convolution, Pooling, Normalization

#### bridge.go

| Function | mlx-c call | Notes |
|----------|-----------|-------|
| `Conv1d(input, kernel, stride, padding, dilation, groups)` | `mlx_conv1d` | |
| `Conv2d(input, kernel, stride, padding, dilation, groups)` | `mlx_conv2d` | |
| `Conv3d(input, kernel, stride, padding, dilation, groups)` | `mlx_conv3d` | 3D conv (MPSGraph doesn't have) |
| `ConvGeneral(...)` | Decompose to conv1d/2d/3d | Map GoMLX ConvGeneral params |

#### function.go — Pooling

MLX doesn't have dedicated pooling ops in mlx-c (they're in the Python `nn` module). Implement via ReduceWindow decomposition:

- **MaxPool** → sliding window with `mlx_max`
- **AvgPool** → sliding window with `mlx_mean`
- **SumPool** → sliding window with `mlx_sum`

Alternatively, use `mlx_fast_metal_kernel()` to dispatch custom pooling if performance is critical.

#### function.go — Normalization

| Op | Implementation |
|----|---------------|
| BatchNormForInference | Manual: `(x - mean) / sqrt(var + eps) * scale + offset` using MLX ops |
| BatchNormForTraining | Manual: compute batch mean/var, then normalize |
| BatchNormGradient | **Use mlx autograd** — this is where MLX shines. Define forward, get gradient automatically |

#### Validation

- [ ] Conv2d with various strides, padding, dilation, groups
- [ ] MaxPool2d, AvgPool2d
- [ ] BatchNorm inference matches MPSGraph results
- [ ] BatchNorm training returns correct running stats

---

### Phase 6: Graph Compilation & Execution

This is where MLX's lazy evaluation maps to GoMLX's build-then-execute model.

#### Key Insight: Deferred vs Eager

**MPSGraph backend**: Operations immediately create MPSGraphTensor nodes in an Objective-C graph. `Compile()` calls `mpsgraph_compile()` to produce an MPSGraphExecutable. `Execute()` runs it.

**MLX backend**: Operations are lazy — calling `bridge.Add(a, b)` returns an `mlx_array` that represents a *deferred* computation. No Metal work happens. We wrap the full graph in an `mlx_closure`, then `mlx_compile()` it for JIT optimization.

#### builder.go

```go
type Builder struct {
    backend  *Backend
    name     string
    mainFn   *Function
    compiled bool
}

func (b *Builder) Compile() (backends.Executable, error) {
    // 1. Collect input placeholders and output arrays from mainFn
    // 2. Wrap as mlx_closure: inputs → outputs
    // 3. Call mlx_compile() for JIT optimization
    // 4. Return Executable wrapping the compiled closure
}
```

#### function.go — Graph Building Model

```go
type graphNode struct {
    array *bridge.Array   // Lazy mlx_array (not yet evaluated)
    shape shapes.Shape
    name  string          // For parameters
    owner *Function
}

type Function struct {
    backend *Backend
    name    string
    parent  *Function
    params  []*graphNode  // Input placeholders
    outputs []*graphNode  // Return values
    // Control flow tracking (same pattern as MPSGraph)
    controlFlowStep *controlFlowStep
}
```

**Parameter creation**: Use `mlx_array_new()` as placeholder. During execution, real data is substituted.

**Alternative approach**: Since MLX is lazy, we can use actual `mlx_array` objects as placeholders and substitute them at execution time via the closure mechanism. The mlx-c `mlx_closure` captures the computation graph implicitly through array dependencies.

#### executable.go

```go
type Executable struct {
    backend      *Backend
    compiled     C.mlx_closure    // JIT-compiled MLX function
    inputShapes  []shapes.Shape
    outputShapes []shapes.Shape
    mu           sync.Mutex
}

func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, device backends.DeviceNum) ([]backends.Buffer, error) {
    // 1. Extract mlx_array handles from input mlxBuffers
    // 2. Pack into mlx_vector_array
    // 3. Call compiled closure: mlx_closure_apply()
    // 4. mlx_eval() on outputs to materialize results
    // 5. Wrap output arrays as mlxBuffers (zero-copy — they're in unified memory)
    return outputs, nil
}
```

**Performance note**: After `mlx_compile()`, subsequent calls reuse the compiled Metal kernel pipeline. This is analogous to MPSGraph's `mpsgraph_compile()` but with additional operation fusion.

#### Control Flow

MLX handles control flow differently from MPSGraph:

- **While**: Go-side loop calling compiled closure per iteration (same as MPSGraph)
- **If**: Go-side branch selection (same as MPSGraph)
- **Sort**: Use `mlx_sort` / `mlx_argsort` directly (no CPU-side comparator needed!)
- **Call**: Direct closure invocation

For While/If, use the same `ExecutableWithCF` pattern from MPSGraph:
1. Compile pre-control-flow subgraph
2. Compile closure bodies separately
3. CPU orchestrates loop/branch, GPU executes each step

#### Validation

- [ ] Simple graph: constant + parameter → add → compile → execute
- [ ] Multi-output graph
- [ ] Graph with shared subexpressions (MLX should optimize via JIT)
- [ ] While loop (simple counter)
- [ ] If conditional
- [ ] Sort operation
- [ ] Verify `mlx_compile()` produces faster execution than uncompiled

---

### Phase 7: Fused Operations

#### function_fused.go

| GoMLX FusedOp | MLX implementation | Notes |
|---------------|-------------------|-------|
| `FusedSoftmax` | `mlx_softmax` | Direct |
| `FusedGelu` | Manual: `x * 0.5 * (1 + erf(x / sqrt(2)))` or tanh approx | Or use mlx-c if available |
| `FusedLayerNorm` | `mlx_fast_layer_norm` | Custom Metal kernel |
| `FusedDense` | `mlx_matmul` + `mlx_add` (bias) + activation | Compose from primitives |
| `FusedScaledDotProductAttention` | `mlx_fast_scaled_dot_product_attention` | **Optimized Metal kernel** |
| `FusedAttentionQKVProjection` | `mlx_matmul` + split | Compose from primitives |

**Additional MLX-specific fast ops** (available via `mlx_fast_*`):
- `mlx_fast_rms_norm` — RMS normalization (common in LLaMA)
- `mlx_fast_rope` / `mlx_fast_rope_dynamic` — Rotary position embeddings
- These don't have GoMLX FusedOp equivalents yet, but can be exposed as backend-specific extensions.

#### Validation

- [ ] Softmax matches reference
- [ ] LayerNorm matches reference (with and without affine params)
- [ ] Scaled dot product attention matches reference
- [ ] RoPE produces correct rotary embeddings

---

### Phase 8: Quantization Support

This is critical for QLoRA finetuning in gopeft.

#### bridge.go — Quantization

| Function | mlx-c call | Purpose |
|----------|-----------|---------|
| `Quantize(a, groupSize, bits)` | `mlx_quantize` | Quantize weights |
| `Dequantize(a, scales, biases, groupSize, bits)` | `mlx_dequantize` | Dequantize weights |
| `QuantizedMatMul(x, w, scales, biases, ...)` | `mlx_quantized_matmul` | Fused quantized matmul |

MLX's `quantized_matmul` is a **single fused Metal kernel** that:
1. Dequantizes weight on-the-fly
2. Performs matrix multiplication
3. Never materializes full-precision weights in memory

This directly supports gopeft's QLoRA which currently does dequant → matmul as separate steps.

#### Exposing to GoMLX

Quantization ops don't exist in the standard `backends.StandardOps` interface. Options:

1. **Backend-specific extension**: Type-assert to `*mlx.Backend` and call quantization methods directly
2. **Propose upstream**: Add quantization ops to GoMLX's `backends.FusedOps`
3. **Use from gopeft directly**: gopeft already does quantization in Go; MLX backend just needs the underlying math ops to be fast

**Recommended**: Option 1 initially, propose option 2 upstream once proven.

---

### Phase 9: Random Number Generation

GoMLX uses `RNGBitGenerator(state, shape) → (newState, values)`. MLX uses a different RNG model with explicit keys.

#### bridge.go — RNG

```go
func RandomKey(seed uint64) *Array { ... }
func RandomSplit(key *Array) (*Array, *Array) { ... }
func RandomUniform(key *Array, shape []int, dtype DType, stream *Stream) *Array { ... }
func RandomNormal(key *Array, shape []int, dtype DType, stream *Stream) *Array { ... }
func RandomBernoulli(key *Array, p *Array, shape []int, stream *Stream) *Array { ... }
```

#### function.go — RNGBitGenerator

Map GoMLX's RNG model to MLX's:
- GoMLX passes an RNG state tensor and expects (newState, randomBits)
- MLX uses key-based RNG with explicit splitting
- Bridge: extract seed from GoMLX state → create MLX key → generate → return new state

This requires careful state management to maintain reproducibility.

---

## Testing Strategy

### Unit Tests (`mlx_test.go`)

Port the comprehensive test suite from `mpsgraph/gomlx/mpsgraph_test.go` (3577 lines). Same test structure:

1. **Backend lifecycle**: Create, finalize, double-finalize
2. **Buffer operations**: Create from flat data, read back, verify
3. **Shared buffers**: Create, write, verify unified memory semantics
4. **Per-operation tests**: Every StandardOps method with reference values
5. **Shape operations**: Reshape, transpose, concat with various ranks
6. **Reductions**: Along different axes, keepdims
7. **DotGeneral**: Contracting, batch, cross axes
8. **Convolution**: Various strides, padding, dilation
9. **Control flow**: While, If, Sort
10. **Fused ops**: Attention, layer norm, softmax
11. **Data types**: Float16, BFloat16, Float64, Int64, Complex64
12. **GoMLX integration**: Use `context.Exec` to verify end-to-end

### Integration Tests

- Run gopeft's existing test suite with `GOMLX_BACKEND=mlx`
- Compare loss curves: MLX vs XLA on same model + data
- Memory usage comparison: MLX (unified) vs MPSGraph (copy)

### Benchmarks

- MatMul throughput: MLX vs MPSGraph vs XLA
- Transformer forward pass latency
- Training step latency (once autograd works)
- Memory footprint during training

---

## gopeft Integration — Future Steps

Once the MLX backend is complete, gopeft needs these changes to leverage it for finetuning:

### 1. Backend Selection

```go
// In gopeft's training code, add MLX as a backend option:
// GOMLX_BACKEND=mlx go run ./e2e/finetune/...
```

No code changes needed — gopeft uses `backends.New()` which respects `GOMLX_BACKEND` env var.

### 2. Quantization Integration

gopeft's `quantization/` package currently implements NF4 dequantization in Go/SIMD (highway). With MLX backend:

```go
// Option A: Use MLX's native quantized matmul (fastest)
if mlxBackend, ok := backend.(*mlx.Backend); ok {
    // Use mlx_quantized_matmul directly
    output = mlxBackend.QuantizedMatMul(input, quantizedWeights, scales, biases, groupSize, bits)
} else {
    // Fall back to current Go-side dequant + matmul
    output = quantization.FusedQuantizedLinear(ctx, input, ...)
}

// Option B: MLX's autograd handles the dequant→matmul chain automatically
// Just build the graph with Dequantize → MatMul and MLX optimizes it
```

### 3. CoreML Hybrid Mode

gopeft already has `coreml/hybrid.go` for using CoreML for inference while training on CPU. With MLX:

```go
// Training on MLX GPU (fast, has autograd)
// Inference/eval on MLX GPU (also fast)
// No need for hybrid mode — MLX does both well
```

### 4. AOT Caching

gopeft's `aot/` module caches compiled XLA executables to avoid 30-60s recompilation. MLX's `mlx_compile()` is fast (no LLVM), so AOT caching is likely unnecessary. But if needed:

- MLX has `mlx_export_function()` / `mlx_imported_function` for serializing compiled functions
- Could be integrated into gopeft's AOT cache mechanism

### 5. Memory Optimization for Large Models

MLX's unified memory enables training larger models than MPSGraph:
- No double-buffering (Go heap + Metal buffer)
- `mlx_set_memory_limit()` / `mlx_set_cache_limit()` for fine-grained control
- `mlx_checkpoint()` for gradient checkpointing (reduce memory during backprop)

gopeft should expose these as training config options:

```go
type TrainingConfig struct {
    // ...existing fields...
    MLXMemoryLimit    int64  // Max unified memory usage (bytes)
    MLXCacheLimit     int64  // Max cache memory (bytes)
    GradientCheckpoint bool  // Enable mlx_checkpoint for memory savings
}
```

### 6. LoRA/QLoRA Finetuning Path

The full finetuning flow with MLX:

```
1. Load base model weights as MLX arrays (unified memory)
2. Quantize base weights if QLoRA: mlx_quantize()
3. Create LoRA adapter matrices (A, B) as trainable MLX arrays
4. Freeze base weights: context.SetTrainable(false)
5. Training loop:
   a. Forward: base_output + scale * (input @ A @ B)
      - For QLoRA: mlx_quantized_matmul for base, regular matmul for LoRA
   b. Loss computation
   c. Backward: GoMLX autograd computes gradients
      - MLX backend executes grad ops on GPU via mlx_value_and_grad
   d. Optimizer step: Adam update on A, B only
      - In-place update on unified memory (no copies!)
6. Save adapter: only A, B matrices (via safetensors)
7. Merge for deployment: base + scale * A @ B
```

### 7. Gradient Computation Architecture

GoMLX computes gradients at the graph level (not the backend level). The backend just needs to execute the gradient graph efficiently. GoMLX's `graph.Gradient()` decomposes each forward op into its backward counterpart. The MLX backend executes these backward ops using the same bridge functions.

However, MLX's native autograd (`mlx_value_and_grad`) could be **more efficient** because it:
- Fuses forward and backward passes
- Shares intermediate activations
- Applies MLX-specific backward kernel optimizations

**Future optimization**: Add a backend-level `ValueAndGrad()` extension that bypasses GoMLX's graph-level autograd and delegates entirely to MLX. This would require:
1. New interface in GoMLX backends (upstream proposal)
2. MLX backend wrapping the forward graph as `mlx_closure`
3. Calling `mlx_value_and_grad()` on the closure
4. Returning both outputs and gradients

This is a **post-v1 optimization** — GoMLX's graph-level autograd works fine initially.

---

## Implementation Order & Dependencies

```
Phase 1: Bridge + Buffer          ──┐
                                    ├─→ Phase 6: Compilation + Execution
Phase 2: Arithmetic Ops           ──┤
                                    │
Phase 3: Shape + Reduction Ops    ──┤
                                    ├─→ Phase 7: Fused Ops
Phase 4: Linear Algebra           ──┘         │
                                              ├─→ Phase 9: RNG ──→ INFERENCE PARITY
Phase 5: Conv + Pool + Norm       ────────────┘

Phase 8: Quantization             ────────────→ QLORA READY (gopeft integration)
```

**Milestones**:
- **M1**: Phase 1-2 complete → basic ops work, can run simple computations
- **M2**: Phase 3-6 complete → GoMLX integration works, can compile and execute graphs
- **M3**: Phase 7-9 complete → inference parity with MPSGraph backend
- **M4**: Phase 8 + gopeft integration → QLoRA finetuning on Apple Silicon

---

## Risk Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **mlx-c build complexity** | Users must build from source | Provide `scripts/build_mlx.sh`; consider vendoring pre-built binaries for arm64 |
| **mlx-c API changes** | Breaking changes between versions | Pin to specific release tag; test in CI |
| **Lazy evaluation semantics** | Unexpected behavior if eval timing is wrong | Always `mlx_eval()` before reading data back to Go; document clearly |
| **Memory management** | Leaks if mlx_array not freed | Use Go finalizers (`runtime.SetFinalizer`) as safety net; explicit free in BufferFinalize |
| **CGo overhead per op** | Many small CGo calls during graph building | MLX is lazy — CGo calls just build graph, no GPU work. JIT compilation amortizes. Benchmark to verify. |
| **GoMLX autograd vs MLX autograd** | Suboptimal gradient computation | GoMLX autograd works fine initially; native MLX autograd is post-v1 optimization |
| **No existing Go+MLX precedent** | Unknown unknowns | mlx-c is designed for FFI (used by mlx-swift); C API is stable and well-documented |
| **Static library size** | MLX + mlx-c could be large | Release builds with LTO; strip debug symbols; ~50MB expected |

---

## Open Questions

1. **Placeholder strategy**: Should we use actual `mlx_array` as parameter placeholders (and substitute via closure capture), or create a separate placeholder mechanism? The closure approach is more idiomatic to MLX.

2. **Stream management**: Should each Builder get its own stream, or share the backend's GPU stream? MLX streams enable concurrent execution on different streams. For training, a single GPU stream is likely optimal to avoid synchronization overhead.

3. **Error handling**: mlx-c uses return codes + global error handler. Should we set a Go-side error handler via `mlx_set_error_handler()` that captures errors, or check return codes per call? Recommend: check return codes (explicit, no global state).

4. **Memory limits**: Should the backend auto-configure `mlx_set_memory_limit()` based on available RAM, or leave it to the user? Recommend: leave default (75% of RAM, MLX's default) but allow override via config string.

5. **Compiled function caching**: MLX caches compiled functions by default. Should we expose `mlx_detail_compile_clear_cache()` through the backend for memory-constrained scenarios?
