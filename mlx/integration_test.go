// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"
	"testing"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// ===========================================================================
// Test helpers
// ===========================================================================

func newBackend(t *testing.T) *Backend {
	t.Helper()
	b, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	return b.(*Backend)
}

func buildAndExec(t *testing.T, b *Backend, name string, inputShapes []shapes.Shape, inputData [][]float32,
	buildFn func(fn backends.Function, params []backends.Value) []backends.Value,
) [][]float32 {
	t.Helper()
	builder := b.Builder(name)
	main := builder.Main()
	fn := main.(backends.Function)

	params := make([]backends.Value, len(inputShapes))
	for i, s := range inputShapes {
		params[i], _ = fn.Parameter("p"+string(rune('0'+i)), s, nil)
	}

	outputs := buildFn(fn, params)
	fn.Return(outputs, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile %s failed: %v", name, err)
	}
	defer exec.Finalize()

	inputs := make([]backends.Buffer, len(inputData))
	for i, data := range inputData {
		inputs[i], err = b.BufferFromFlatData(0, data, inputShapes[i])
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
	}

	results, err := exec.Execute(inputs, nil, 0)
	if err != nil {
		t.Fatalf("Execute %s failed: %v", name, err)
	}

	outShapes := exec.Outputs()
	outData := make([][]float32, len(results))
	for i, buf := range results {
		size := outShapes[i].Size()
		outData[i] = make([]float32, size)
		if err := b.BufferToFlatData(buf, outData[i]); err != nil {
			t.Fatalf("BufferToFlatData output %d failed: %v", i, err)
		}
	}
	return outData
}

func assertClose(t *testing.T, name string, got, expected []float32, tol float64) {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("%s: length mismatch: got %d, expected %d", name, len(got), len(expected))
	}
	for i, g := range got {
		if math.Abs(float64(g-expected[i])) > tol {
			t.Errorf("%s[%d]: expected %f, got %f", name, i, expected[i], g)
		}
	}
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func TestUnaryOps(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	input := []float32{1, 4, 9, 16}

	tests := []struct {
		name     string
		opFn     func(fn backends.Function, x backends.Value) backends.Value
		expected []float32
		tol      float64
	}{
		{"Abs", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Abs(x); return v }, []float32{1, 4, 9, 16}, 1e-5},
		{"Neg", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Neg(x); return v }, []float32{-1, -4, -9, -16}, 1e-5},
		{"Sqrt", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Sqrt(x); return v }, []float32{1, 2, 3, 4}, 1e-5},
		{"Exp", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Exp(x); return v },
			[]float32{float32(math.Exp(1)), float32(math.Exp(4)), float32(math.Exp(9)), float32(math.Exp(16))}, 1e-2},
		{"Log", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Log(x); return v },
			[]float32{0, float32(math.Log(4)), float32(math.Log(9)), float32(math.Log(16))}, 1e-5},
		{"Floor", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Floor(x); return v }, []float32{1, 4, 9, 16}, 1e-5},
		{"Ceil", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Ceil(x); return v }, []float32{1, 4, 9, 16}, 1e-5},
		{"Sign", func(fn backends.Function, x backends.Value) backends.Value { v, _ := fn.Sign(x); return v }, []float32{1, 1, 1, 1}, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := buildAndExec(t, b, tt.name, []shapes.Shape{shape}, [][]float32{input},
				func(fn backends.Function, params []backends.Value) []backends.Value {
					return []backends.Value{tt.opFn(fn, params[0])}
				})
			assertClose(t, tt.name, results[0], tt.expected, tt.tol)
		})
	}
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func TestBinaryOps(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhs := []float32{10, 20, 30, 40}
	rhs := []float32{3, 4, 5, 8}

	tests := []struct {
		name     string
		opFn     func(fn backends.Function, l, r backends.Value) backends.Value
		expected []float32
	}{
		{"Add", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Add(l, r); return v }, []float32{13, 24, 35, 48}},
		{"Sub", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Sub(l, r); return v }, []float32{7, 16, 25, 32}},
		{"Mul", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Mul(l, r); return v }, []float32{30, 80, 150, 320}},
		{"Div", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Div(l, r); return v }, []float32{10.0 / 3, 5, 6, 5}},
		{"Max", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Max(l, r); return v }, []float32{10, 20, 30, 40}},
		{"Min", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Min(l, r); return v }, []float32{3, 4, 5, 8}},
		{"Pow", func(fn backends.Function, l, r backends.Value) backends.Value { v, _ := fn.Pow(l, r); return v }, []float32{float32(math.Pow(10, 3)), float32(math.Pow(20, 4)), float32(math.Pow(30, 5)), float32(math.Pow(40, 8))}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := buildAndExec(t, b, tt.name,
				[]shapes.Shape{shape, shape}, [][]float32{lhs, rhs},
				func(fn backends.Function, params []backends.Value) []backends.Value {
					return []backends.Value{tt.opFn(fn, params[0], params[1])}
				})
			assertClose(t, tt.name, results[0], tt.expected, 100.0)
		})
	}
}

// ===========================================================================
// Shape Operations
// ===========================================================================

func TestReshape(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	input := []float32{1, 2, 3, 4, 5, 6}
	results := buildAndExec(t, b, "Reshape",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 2, 3)},
		[][]float32{input},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Reshape(params[0], 3, 2)
			return []backends.Value{v}
		})
	assertClose(t, "Reshape", results[0], input, 1e-5)
}

func TestTranspose(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
	results := buildAndExec(t, b, "Transpose",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 2, 3)},
		[][]float32{{1, 2, 3, 4, 5, 6}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Transpose(params[0], 1, 0)
			return []backends.Value{v}
		})
	assertClose(t, "Transpose", results[0], []float32{1, 4, 2, 5, 3, 6}, 1e-5)
}

func TestSlice(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// [1,2,3,4,5,6] -> [2,3,4]
	results := buildAndExec(t, b, "Slice",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 6)},
		[][]float32{{1, 2, 3, 4, 5, 6}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Slice(params[0], []int{1}, []int{4}, []int{1})
			return []backends.Value{v}
		})
	assertClose(t, "Slice", results[0], []float32{2, 3, 4}, 1e-5)
}

func TestConcatenate(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 3)
	results := buildAndExec(t, b, "Concatenate",
		[]shapes.Shape{shape, shape},
		[][]float32{{1, 2, 3}, {4, 5, 6}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Concatenate(0, params[0], params[1])
			return []backends.Value{v}
		})
	assertClose(t, "Concatenate", results[0], []float32{1, 2, 3, 4, 5, 6}, 1e-5)
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

func TestReductions(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// [[1,2,3],[4,5,6]] -> reduce along axis 1
	shape := shapes.Make(dtypes.Float32, 2, 3)
	input := []float32{1, 2, 3, 4, 5, 6}

	t.Run("ReduceSum", func(t *testing.T) {
		results := buildAndExec(t, b, "ReduceSum",
			[]shapes.Shape{shape}, [][]float32{input},
			func(fn backends.Function, params []backends.Value) []backends.Value {
				v, _ := fn.ReduceSum(params[0], 1)
				return []backends.Value{v}
			})
		assertClose(t, "ReduceSum", results[0], []float32{6, 15}, 1e-5)
	})

	t.Run("ReduceMax", func(t *testing.T) {
		results := buildAndExec(t, b, "ReduceMax",
			[]shapes.Shape{shape}, [][]float32{input},
			func(fn backends.Function, params []backends.Value) []backends.Value {
				v, _ := fn.ReduceMax(params[0], 1)
				return []backends.Value{v}
			})
		assertClose(t, "ReduceMax", results[0], []float32{3, 6}, 1e-5)
	})

	t.Run("ReduceMin", func(t *testing.T) {
		results := buildAndExec(t, b, "ReduceMin",
			[]shapes.Shape{shape}, [][]float32{input},
			func(fn backends.Function, params []backends.Value) []backends.Value {
				v, _ := fn.ReduceMin(params[0], 1)
				return []backends.Value{v}
			})
		assertClose(t, "ReduceMin", results[0], []float32{1, 4}, 1e-5)
	})

	t.Run("ReduceProduct", func(t *testing.T) {
		results := buildAndExec(t, b, "ReduceProduct",
			[]shapes.Shape{shape}, [][]float32{input},
			func(fn backends.Function, params []backends.Value) []backends.Value {
				v, _ := fn.ReduceProduct(params[0], 1)
				return []backends.Value{v}
			})
		assertClose(t, "ReduceProduct", results[0], []float32{6, 120}, 1e-5)
	})
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

func TestDot(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// [2,3] @ [3,2] = [2,2]
	results := buildAndExec(t, b, "Dot",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 2, 3), shapes.Make(dtypes.Float32, 3, 2)},
		[][]float32{{1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.(*Function).Dot(params[0], params[1])
			return []backends.Value{v}
		})
	assertClose(t, "Dot", results[0], []float32{14, 32, 32, 77}, 1e-4)
}

// ===========================================================================
// Constant Values
// ===========================================================================

func TestConstant(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// z = x + [10, 20, 30]
	shape := shapes.Make(dtypes.Float32, 3)
	results := buildAndExec(t, b, "Constant",
		[]shapes.Shape{shape},
		[][]float32{{1, 2, 3}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			c, _ := fn.Constant([]float32{10, 20, 30}, 3)
			v, _ := fn.Add(params[0], c)
			return []backends.Value{v}
		})
	assertClose(t, "Constant", results[0], []float32{11, 22, 33}, 1e-5)
}

// ===========================================================================
// ConvertDType
// ===========================================================================

func TestConvertDType(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Convert float32 -> float32 (identity) and verify values preserved.
	shape := shapes.Make(dtypes.Float32, 3)
	results := buildAndExec(t, b, "ConvertDType",
		[]shapes.Shape{shape},
		[][]float32{{1.5, 2.7, 3.9}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			// Convert to int32 then back to float32 (truncates).
			v1, _ := fn.ConvertDType(params[0], dtypes.Int32)
			v2, _ := fn.ConvertDType(v1, dtypes.Float32)
			return []backends.Value{v2}
		})
	assertClose(t, "ConvertDType", results[0], []float32{1, 2, 3}, 1e-5)
}

// ===========================================================================
// Where
// ===========================================================================

func TestWhere(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// x > 2 ? x : 0
	shape := shapes.Make(dtypes.Float32, 4)
	results := buildAndExec(t, b, "Where",
		[]shapes.Shape{shape},
		[][]float32{{1, 2, 3, 4}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			two, _ := fn.Constant([]float32{2, 2, 2, 2}, 4)
			cond, _ := fn.GreaterThan(params[0], two)
			zero, _ := fn.Constant([]float32{0, 0, 0, 0}, 4)
			v, _ := fn.Where(cond, params[0], zero)
			return []backends.Value{v}
		})
	assertClose(t, "Where", results[0], []float32{0, 0, 3, 4}, 1e-5)
}

// ===========================================================================
// Fused Operations
// ===========================================================================

func TestFusedSoftmax(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	results := buildAndExec(t, b, "Softmax",
		[]shapes.Shape{shape},
		[][]float32{{1, 2, 3, 4}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.FusedSoftmax(params[0], 0)
			return []backends.Value{v}
		})
	// Verify sum ≈ 1.
	sum := float32(0)
	for _, v := range results[0] {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum: expected 1.0, got %f", sum)
	}
	// Verify monotonically increasing.
	for i := 1; i < len(results[0]); i++ {
		if results[0][i] <= results[0][i-1] {
			t.Errorf("Softmax not monotonic: [%d]=%f <= [%d]=%f", i, results[0][i], i-1, results[0][i-1])
		}
	}
}

// ===========================================================================
// Multiple Outputs
// ===========================================================================

func TestMultipleOutputs(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 3)
	results := buildAndExec(t, b, "MultiOutput",
		[]shapes.Shape{shape, shape},
		[][]float32{{1, 2, 3}, {10, 20, 30}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			sum, _ := fn.Add(params[0], params[1])
			diff, _ := fn.Sub(params[0], params[1])
			return []backends.Value{sum, diff}
		})
	assertClose(t, "sum", results[0], []float32{11, 22, 33}, 1e-5)
	assertClose(t, "diff", results[1], []float32{-9, -18, -27}, 1e-5)
}

// ===========================================================================
// Re-execution (Step 6: same Executable, different inputs)
// ===========================================================================

func TestReExecution(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Build z = x * y + x
	shape := shapes.Make(dtypes.Float32, 3)
	builder := b.Builder("reexec")
	main := builder.Main()
	fn := main.(backends.Function)
	x, _ := fn.Parameter("x", shape, nil)
	y, _ := fn.Parameter("y", shape, nil)
	xy, _ := fn.Mul(x, y)
	z, _ := fn.Add(xy, x)
	fn.Return([]backends.Value{z}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// Execute multiple times with different inputs.
	testCases := []struct {
		xData, yData []float32
		expected     []float32
	}{
		{[]float32{1, 2, 3}, []float32{2, 3, 4}, []float32{3, 8, 15}},
		{[]float32{0, 0, 0}, []float32{5, 5, 5}, []float32{0, 0, 0}},
		{[]float32{-1, -2, -3}, []float32{1, 1, 1}, []float32{-2, -4, -6}},
		{[]float32{10, 20, 30}, []float32{0.5, 0.5, 0.5}, []float32{15, 30, 45}},
	}

	for i, tc := range testCases {
		xBuf, _ := b.BufferFromFlatData(0, tc.xData, shape)
		yBuf, _ := b.BufferFromFlatData(0, tc.yData, shape)

		results, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute #%d failed: %v", i, err)
		}

		result := make([]float32, 3)
		b.BufferToFlatData(results[0], result)
		assertClose(t, "ReExec", result, tc.expected, 1e-4)
	}
}

// ===========================================================================
// Stress Test: many repeated executions (memory management)
// ===========================================================================

func TestStressRepeatedExecution(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Build a moderately complex graph: softmax(x * W + b)
	xShape := shapes.Make(dtypes.Float32, 4, 8)
	wShape := shapes.Make(dtypes.Float32, 8, 4)
	bShape := shapes.Make(dtypes.Float32, 4)

	builder := b.Builder("stress")
	main := builder.Main()
	fn := main.(backends.Function)
	xp, _ := fn.Parameter("x", xShape, nil)
	wp, _ := fn.Parameter("w", wShape, nil)
	bp, _ := fn.Parameter("b", bShape, nil)
	mm, _ := fn.(*Function).Dot(xp, wp) // [4,4]
	// Broadcast bias [4] -> [4,4]
	bBroadcast, _ := fn.BroadcastInDim(bp, shapes.Make(dtypes.Float32, 4, 4), []int{1})
	added, _ := fn.Add(mm, bBroadcast)
	sm, _ := fn.FusedSoftmax(added, 1)
	fn.Return([]backends.Value{sm}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data.
	xData := make([]float32, 32)
	wData := make([]float32, 32)
	bData := make([]float32, 4)
	for i := range xData {
		xData[i] = float32(i) * 0.1
	}
	for i := range wData {
		wData[i] = float32(i) * 0.01
	}
	for i := range bData {
		bData[i] = 0.1
	}

	// Execute 100 times — should not leak memory or crash.
	for iter := 0; iter < 100; iter++ {
		xBuf, _ := b.BufferFromFlatData(0, xData, xShape)
		wBuf, _ := b.BufferFromFlatData(0, wData, wShape)
		bBuf, _ := b.BufferFromFlatData(0, bData, bShape)

		results, err := exec.Execute([]backends.Buffer{xBuf, wBuf, bBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute iteration %d failed: %v", iter, err)
		}

		// Verify each row of softmax sums to 1.
		result := make([]float32, 16) // 4x4
		b.BufferToFlatData(results[0], result)
		for row := 0; row < 4; row++ {
			sum := float32(0)
			for col := 0; col < 4; col++ {
				sum += result[row*4+col]
			}
			if math.Abs(float64(sum-1.0)) > 1e-4 {
				t.Fatalf("Iteration %d, row %d: softmax sum = %f, expected 1.0", iter, row, sum)
			}
		}
	}
}

// ===========================================================================
// Chain of operations (deep graph)
// ===========================================================================

func TestDeepGraph(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	results := buildAndExec(t, b, "DeepGraph",
		[]shapes.Shape{shape},
		[][]float32{{1, 2, 3, 4}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v := params[0]
			// Chain: (((x + x) * 0.5) - 1) + 2 = x + 1
			added, _ := fn.Add(v, v)                                   // 2x
			half, _ := fn.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
			halved, _ := fn.Mul(added, half)                            // x
			one, _ := fn.Constant([]float32{1, 1, 1, 1}, 4)
			subbed, _ := fn.Sub(halved, one)                            // x - 1
			two, _ := fn.Constant([]float32{2, 2, 2, 2}, 4)
			result, _ := fn.Add(subbed, two)                            // x + 1
			return []backends.Value{result}
		})
	assertClose(t, "DeepGraph", results[0], []float32{2, 3, 4, 5}, 1e-5)
}

// ===========================================================================
// Clamp
// ===========================================================================

func TestClamp(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	shape := shapes.Make(dtypes.Float32, 6)
	results := buildAndExec(t, b, "Clamp",
		[]shapes.Shape{shape},
		[][]float32{{-5, -1, 0, 3, 7, 10}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			lo, _ := fn.Constant([]float32{0, 0, 0, 0, 0, 0}, 6)
			hi, _ := fn.Constant([]float32{5, 5, 5, 5, 5, 5}, 6)
			v, _ := fn.Clamp(lo, params[0], hi)
			return []backends.Value{v}
		})
	assertClose(t, "Clamp", results[0], []float32{0, 0, 0, 3, 5, 5}, 1e-5)
}

// ===========================================================================
// Reverse
// ===========================================================================

func TestReverse(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	results := buildAndExec(t, b, "Reverse",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 6)},
		[][]float32{{1, 2, 3, 4, 5, 6}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Reverse(params[0], 0)
			return []backends.Value{v}
		})
	assertClose(t, "Reverse", results[0], []float32{6, 5, 4, 3, 2, 1}, 1e-5)
}

// ===========================================================================
// BroadcastInDim
// ===========================================================================

func TestBroadcastInDim(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Broadcast [3] -> [2,3]: repeat along axis 0
	results := buildAndExec(t, b, "BroadcastInDim",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 3)},
		[][]float32{{1, 2, 3}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			outShape := shapes.Make(dtypes.Float32, 2, 3)
			v, _ := fn.BroadcastInDim(params[0], outShape, []int{1})
			return []backends.Value{v}
		})
	assertClose(t, "BroadcastInDim", results[0], []float32{1, 2, 3, 1, 2, 3}, 1e-5)
}

// ===========================================================================
// Identity (value passthrough)
// ===========================================================================

func TestIdentity(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	results := buildAndExec(t, b, "Identity",
		[]shapes.Shape{shapes.Make(dtypes.Float32, 3)},
		[][]float32{{42, 43, 44}},
		func(fn backends.Function, params []backends.Value) []backends.Value {
			v, _ := fn.Identity(params[0])
			return []backends.Value{v}
		})
	assertClose(t, "Identity", results[0], []float32{42, 43, 44}, 1e-5)
}

// TestMLPEndToEnd simulates a 2-layer MLP forward pass with ReLU activation,
// exercising matmul, bias add, relu, softmax, and tape replay with multiple inputs.
func TestMLPEndToEnd(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Layer shapes: input [2,3], W1 [3,4], b1 [4], W2 [4,2], b2 [2]
	xShape := shapes.Make(dtypes.Float32, 2, 3)
	w1Shape := shapes.Make(dtypes.Float32, 3, 4)
	b1Shape := shapes.Make(dtypes.Float32, 4)
	w2Shape := shapes.Make(dtypes.Float32, 4, 2)
	b2Shape := shapes.Make(dtypes.Float32, 2)

	builder := b.Builder("MLP")
	main := builder.Main()
	fn := main.(backends.Function)

	xParam, _ := fn.Parameter("x", xShape, nil)
	w1Param, _ := fn.Parameter("w1", w1Shape, nil)
	b1Param, _ := fn.Parameter("b1", b1Shape, nil)
	w2Param, _ := fn.Parameter("w2", w2Shape, nil)
	b2Param, _ := fn.Parameter("b2", b2Shape, nil)

	// Layer 1: FusedDense with ReLU (matmul + bias + relu)
	h, _ := fn.FusedDense(xParam, w1Param, b1Param, backends.ActivationRelu)

	// Layer 2: FusedDense with no activation, then softmax
	out, _ := fn.FusedDense(h, w2Param, b2Param, backends.ActivationNone)
	out, _ = fn.FusedSoftmax(out, 1)

	fn.Return([]backends.Value{out}, nil)
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// Weights (fixed for reproducibility)
	w1Data := []float32{0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3, 0.3, -0.1, 0.2, 0.1}
	b1Data := []float32{0.01, -0.01, 0.02, -0.02}
	w2Data := []float32{0.2, -0.1, -0.3, 0.4, 0.1, 0.2, -0.2, 0.3}
	b2Data := []float32{0.0, 0.0}

	w1Buf, _ := b.BufferFromFlatData(0, w1Data, w1Shape)
	b1Buf, _ := b.BufferFromFlatData(0, b1Data, b1Shape)
	w2Buf, _ := b.BufferFromFlatData(0, w2Data, w2Shape)
	b2Buf, _ := b.BufferFromFlatData(0, b2Data, b2Shape)

	// Run with two different inputs to verify tape replay.
	for run, xData := range [][]float32{
		{1, 2, 3, 4, 5, 6},
		{0.5, -1, 2, 3, 0, -0.5},
	} {
		xBuf, _ := b.BufferFromFlatData(0, xData, xShape)
		results, err := exec.Execute(
			[]backends.Buffer{xBuf, w1Buf, b1Buf, w2Buf, b2Buf}, nil, 0)
		if err != nil {
			t.Fatalf("Run %d: Execute failed: %v", run, err)
		}

		outData := make([]float32, 4)
		if err := b.BufferToFlatData(results[0], outData); err != nil {
			t.Fatalf("Run %d: BufferToFlatData failed: %v", run, err)
		}

		// Verify softmax properties: each row sums to 1, all values in [0,1].
		for row := 0; row < 2; row++ {
			sum := float64(outData[row*2]) + float64(outData[row*2+1])
			if math.Abs(sum-1.0) > 1e-5 {
				t.Errorf("Run %d row %d: softmax sum = %f, want 1.0", run, row, sum)
			}
			for col := 0; col < 2; col++ {
				v := outData[row*2+col]
				if v < 0 || v > 1 {
					t.Errorf("Run %d row %d col %d: softmax value %f out of [0,1]", run, row, col, v)
				}
			}
		}
		t.Logf("Run %d: MLP output = %v", run, outData)
	}
}

// ===========================================================================
// Autograd Transforms
// ===========================================================================

// TestValueAndGrad tests that mlx_value_and_grad computes correct gradients.
// f(x) = x^2, f'(x) = 2x. For x=3.0, f(x)=9.0, f'(x)=6.0.
func TestValueAndGrad(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	// Create a closure that computes x^2 (sum to get scalar loss).
	squareFn := bridge.NewClosureFromGoFunc(func(inputs []*bridge.Array) []*bridge.Array {
		s := b.stream()
		sq := bridge.Multiply(inputs[0], inputs[0], s)
		// Sum to scalar (value_and_grad requires scalar output).
		summed := bridge.Sum(sq, []int{0}, false, s)
		return []*bridge.Array{summed}
	})
	defer squareFn.Free()

	// Create value_and_grad w.r.t. argument 0.
	vg := bridge.ValueAndGrad(squareFn, []int{0})
	defer vg.Free()

	// Apply with x = [3.0].
	x := bridge.NewArrayFromData(nil, []int{1}, bridge.DTypeFloat32)
	// Create from flat data.
	xData := []float32{3.0}
	x = bridge.NewArrayFromData(unsafe.Pointer(&xData[0]), []int{1}, bridge.DTypeFloat32)

	values, grads, err := vg.Apply([]*bridge.Array{x})
	if err != nil {
		t.Fatalf("ValueAndGrad.Apply failed: %v", err)
	}
	if err := bridge.Eval(append(values, grads...)...); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	// Check value: x^2 = 9.0.
	valPtr := values[0].DataPtr()
	val := *(*float32)(valPtr)
	if math.Abs(float64(val)-9.0) > 0.001 {
		t.Errorf("value: expected 9.0, got %f", val)
	}

	// Check gradient: 2*x = 6.0.
	gradPtr := grads[0].DataPtr()
	grad := *(*float32)(gradPtr)
	if math.Abs(float64(grad)-6.0) > 0.001 {
		t.Errorf("gradient: expected 6.0, got %f", grad)
	}

	t.Logf("f(3.0) = %f, f'(3.0) = %f", val, grad)

	for _, a := range values { a.Free() }
	for _, a := range grads { a.Free() }
	x.Free()
}

// TestVJP tests that mlx_vjp (reverse-mode autodiff) works correctly.
// f(x) = x^2, VJP with cotangent=1 gives gradient = 2*x.
func TestVJP(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	squareFn := bridge.NewClosureFromGoFunc(func(inputs []*bridge.Array) []*bridge.Array {
		s := b.stream()
		sq := bridge.Multiply(inputs[0], inputs[0], s)
		return []*bridge.Array{sq}
	})
	defer squareFn.Free()

	// x = [3.0]
	xData := []float32{3.0}
	x := bridge.NewArrayFromData(unsafe.Pointer(&xData[0]), []int{1}, bridge.DTypeFloat32)

	// cotangent = [1.0] (seed for backward pass)
	cotData := []float32{1.0}
	cot := bridge.NewArrayFromData(unsafe.Pointer(&cotData[0]), []int{1}, bridge.DTypeFloat32)

	outputs, vjps, err := bridge.VJP(squareFn, []*bridge.Array{x}, []*bridge.Array{cot})
	if err != nil {
		t.Fatalf("VJP failed: %v", err)
	}
	if err := bridge.Eval(append(outputs, vjps...)...); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	// Output: x^2 = 9.0.
	outVal := *(*float32)(outputs[0].DataPtr())
	if math.Abs(float64(outVal)-9.0) > 0.001 {
		t.Errorf("output: expected 9.0, got %f", outVal)
	}

	// VJP: d(x^2)/dx * cotangent = 2*3 * 1 = 6.0.
	vjpVal := *(*float32)(vjps[0].DataPtr())
	if math.Abs(float64(vjpVal)-6.0) > 0.001 {
		t.Errorf("vjp: expected 6.0, got %f", vjpVal)
	}

	t.Logf("f(3.0) = %f, vjp = %f", outVal, vjpVal)

	for _, a := range outputs { a.Free() }
	for _, a := range vjps { a.Free() }
	x.Free()
	cot.Free()
}

// TestJVP tests that mlx_jvp (forward-mode autodiff) works correctly.
// f(x) = x^2, JVP with tangent=1 gives directional derivative = 2*x.
func TestJVP(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	squareFn := bridge.NewClosureFromGoFunc(func(inputs []*bridge.Array) []*bridge.Array {
		s := b.stream()
		sq := bridge.Multiply(inputs[0], inputs[0], s)
		return []*bridge.Array{sq}
	})
	defer squareFn.Free()

	// x = [3.0]
	xData := []float32{3.0}
	x := bridge.NewArrayFromData(unsafe.Pointer(&xData[0]), []int{1}, bridge.DTypeFloat32)

	// tangent = [1.0]
	tanData := []float32{1.0}
	tan := bridge.NewArrayFromData(unsafe.Pointer(&tanData[0]), []int{1}, bridge.DTypeFloat32)

	outputs, jvps, err := bridge.JVP(squareFn, []*bridge.Array{x}, []*bridge.Array{tan})
	if err != nil {
		t.Fatalf("JVP failed: %v", err)
	}
	if err := bridge.Eval(append(outputs, jvps...)...); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	// Output: x^2 = 9.0.
	outVal := *(*float32)(outputs[0].DataPtr())
	if math.Abs(float64(outVal)-9.0) > 0.001 {
		t.Errorf("output: expected 9.0, got %f", outVal)
	}

	// JVP: d(x^2)/dx * tangent = 2*3 * 1 = 6.0.
	jvpVal := *(*float32)(jvps[0].DataPtr())
	if math.Abs(float64(jvpVal)-6.0) > 0.001 {
		t.Errorf("jvp: expected 6.0, got %f", jvpVal)
	}

	t.Logf("f(3.0) = %f, jvp = %f", outVal, jvpVal)

	for _, a := range outputs { a.Free() }
	for _, a := range jvps { a.Free() }
	x.Free()
	tan.Free()
}

// TestValueAndGradMultiParam tests value_and_grad with multiple parameters.
// f(x, y) = sum(x * y), df/dx = y, df/dy = x.
func TestValueAndGradMultiParam(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	dotFn := bridge.NewClosureFromGoFunc(func(inputs []*bridge.Array) []*bridge.Array {
		s := b.stream()
		prod := bridge.Multiply(inputs[0], inputs[1], s)
		summed := bridge.Sum(prod, []int{0}, false, s)
		return []*bridge.Array{summed}
	})
	defer dotFn.Free()

	// Differentiate w.r.t. both arguments.
	vg := bridge.ValueAndGrad(dotFn, []int{0, 1})
	defer vg.Free()

	// x = [2.0, 3.0], y = [4.0, 5.0]
	xData := []float32{2.0, 3.0}
	yData := []float32{4.0, 5.0}
	x := bridge.NewArrayFromData(unsafe.Pointer(&xData[0]), []int{2}, bridge.DTypeFloat32)
	y := bridge.NewArrayFromData(unsafe.Pointer(&yData[0]), []int{2}, bridge.DTypeFloat32)

	values, grads, err := vg.Apply([]*bridge.Array{x, y})
	if err != nil {
		t.Fatalf("ValueAndGrad.Apply failed: %v", err)
	}
	if err := bridge.Eval(append(values, grads...)...); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	// Value: sum(x*y) = 2*4 + 3*5 = 23.0.
	val := *(*float32)(values[0].DataPtr())
	if math.Abs(float64(val)-23.0) > 0.001 {
		t.Errorf("value: expected 23.0, got %f", val)
	}

	// Grad w.r.t. x = y = [4.0, 5.0].
	gradX0 := *(*float32)(grads[0].DataPtr())
	gradX := make([]float32, 2)
	gradX[0] = gradX0
	gradXArr := grads[0]
	size := gradXArr.Size()
	if size != 2 {
		t.Fatalf("grad x size: expected 2, got %d", size)
	}

	// Grad w.r.t. y = x = [2.0, 3.0].
	gradYArr := grads[1]
	sizeY := gradYArr.Size()
	if sizeY != 2 {
		t.Fatalf("grad y size: expected 2, got %d", sizeY)
	}

	t.Logf("f(x,y) = %f, grad_x size = %d, grad_y size = %d", val, size, sizeY)

	for _, a := range values { a.Free() }
	for _, a := range grads { a.Free() }
	x.Free()
	y.Free()
}

// ===========================================================================
// Memory Profiling Tests
// ===========================================================================

func TestMemoryProfiling(t *testing.T) {
	b := newBackend(t)
	defer b.Finalize()

	bridge.ResetPeakMemory()

	// Allocate a buffer to force some memory usage.
	data := make([]float32, 1024)
	shape := shapes.Make(dtypes.Float32, 1024)
	buf, err := b.BufferFromFlatData(0, data, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %v", err)
	}
	_ = buf

	peak := bridge.GetPeakMemory()
	cache := bridge.GetCacheMemory()
	active := bridge.GetActiveMemory()

	t.Logf("Peak: %d bytes, Cache: %d bytes, Active: %d bytes", peak, cache, active)

	// SetWiredLimit should return a value.
	prev := bridge.SetWiredLimit(1 << 30) // 1GB
	t.Logf("Previous wired limit: %d", prev)
}

// ===========================================================================
// Cumulative Ops Tests
// ===========================================================================

func TestCumSum(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{1, 2, 3, 4}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{4}, bridge.DTypeFloat32)

	result := bridge.CumSum(x, 0, false, true, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	out := make([]float32, 4)
	ptr := result.DataPtr()
	copy(out, unsafe.Slice((*float32)(ptr), 4))

	expected := []float32{1, 3, 6, 10}
	for i, v := range out {
		if math.Abs(float64(v)-float64(expected[i])) > 0.001 {
			t.Errorf("cumsum[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
	t.Logf("CumSum: %v", out)
	result.Free()
	x.Free()
}

func TestCumProd(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{1, 2, 3, 4}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{4}, bridge.DTypeFloat32)

	result := bridge.CumProd(x, 0, false, true, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	out := make([]float32, 4)
	copy(out, unsafe.Slice((*float32)(result.DataPtr()), 4))

	expected := []float32{1, 2, 6, 24}
	for i, v := range out {
		if math.Abs(float64(v)-float64(expected[i])) > 0.001 {
			t.Errorf("cumprod[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
	t.Logf("CumProd: %v", out)
	result.Free()
	x.Free()
}

// ===========================================================================
// Statistical Ops Tests
// ===========================================================================

func TestMean(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{1, 2, 3, 4, 5, 6}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2, 3}, bridge.DTypeFloat32)

	// Mean over all elements.
	result := bridge.Mean(x, false, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	val := *(*float32)(result.DataPtr())
	if math.Abs(float64(val)-3.5) > 0.001 {
		t.Errorf("mean: expected 3.5, got %f", val)
	}
	t.Logf("Mean(all): %f", val)

	// Mean along axis 1.
	result2 := bridge.MeanAxis(x, 1, false, s)
	if err := bridge.Eval(result2); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	out := make([]float32, 2)
	copy(out, unsafe.Slice((*float32)(result2.DataPtr()), 2))
	if math.Abs(float64(out[0])-2.0) > 0.001 || math.Abs(float64(out[1])-5.0) > 0.001 {
		t.Errorf("mean(axis=1): expected [2.0, 5.0], got %v", out)
	}
	t.Logf("Mean(axis=1): %v", out)

	result.Free()
	result2.Free()
	x.Free()
}

func TestVarianceAndStd(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{2, 4, 4, 4, 5, 5, 7, 9}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{8}, bridge.DTypeFloat32)

	// Population variance (ddof=0).
	vResult := bridge.Variance(x, false, 0, s)
	if err := bridge.Eval(vResult); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	variance := *(*float32)(vResult.DataPtr())
	if math.Abs(float64(variance)-4.0) > 0.01 {
		t.Errorf("variance: expected 4.0, got %f", variance)
	}

	// Std dev.
	sResult := bridge.StdDev(x, false, 0, s)
	if err := bridge.Eval(sResult); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	stddev := *(*float32)(sResult.DataPtr())
	if math.Abs(float64(stddev)-2.0) > 0.01 {
		t.Errorf("std: expected 2.0, got %f", stddev)
	}
	t.Logf("Var: %f, Std: %f", variance, stddev)

	vResult.Free()
	sResult.Free()
	x.Free()
}

func TestLogSumExp(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{1, 2, 3}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{3}, bridge.DTypeFloat32)

	result := bridge.LogSumExp(x, false, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	val := *(*float32)(result.DataPtr())
	// log(e^1 + e^2 + e^3) ≈ 3.4076
	expected := math.Log(math.Exp(1) + math.Exp(2) + math.Exp(3))
	if math.Abs(float64(val)-expected) > 0.01 {
		t.Errorf("logsumexp: expected %f, got %f", expected, val)
	}
	t.Logf("LogSumExp: %f", val)
	result.Free()
	x.Free()
}

// ===========================================================================
// Array Manipulation Tests
// ===========================================================================

func TestTopK(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := []float32{3, 1, 4, 1, 5, 9, 2, 6}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{8}, bridge.DTypeFloat32)

	result := bridge.TopK(x, 3, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	if result.Size() != 3 {
		t.Errorf("topk size: expected 3, got %d", result.Size())
	}
	out := make([]float32, 3)
	copy(out, unsafe.Slice((*float32)(result.DataPtr()), 3))
	t.Logf("TopK(3): %v", out)

	result.Free()
	x.Free()
}

func TestFlatten(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2, 3, 4}, bridge.DTypeFloat32)

	result := bridge.Flatten(x, 1, 2, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 12 {
		t.Errorf("flatten shape: expected [2, 12], got %v", shape)
	}
	t.Logf("Flatten: shape=%v", shape)
	result.Free()
	x.Free()
}

func TestSplit(t *testing.T) {
	s := bridge.DefaultGPUStream()

	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{12}, bridge.DTypeFloat32)

	result := bridge.Split(x, 3, 0, s)
	defer result.Free()
	if result.Size() != 3 {
		t.Errorf("split: expected 3 parts, got %d", result.Size())
	}
	for i := 0; i < result.Size(); i++ {
		part := result.Get(i)
		if part.Size() != 4 {
			t.Errorf("split part %d: expected 4 elements, got %d", i, part.Size())
		}
		part.Free()
	}
	t.Logf("Split into %d parts", result.Size())
	x.Free()
}

func TestLinspace(t *testing.T) {
	s := bridge.DefaultGPUStream()

	result := bridge.Linspace(0, 1, 5, bridge.DTypeFloat32, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	out := make([]float32, 5)
	copy(out, unsafe.Slice((*float32)(result.DataPtr()), 5))
	expected := []float32{0, 0.25, 0.5, 0.75, 1.0}
	for i, v := range out {
		if math.Abs(float64(v)-float64(expected[i])) > 0.001 {
			t.Errorf("linspace[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
	t.Logf("Linspace: %v", out)
	result.Free()
}

// ===========================================================================
// Linear Algebra Tests
// ===========================================================================

func TestLinalgInv(t *testing.T) {
	s := bridge.DefaultCPUStream() // linalg ops require CPU stream

	// 2x2 identity matrix — its inverse is itself.
	data := []float32{1, 0, 0, 1}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2, 2}, bridge.DTypeFloat32)

	inv := bridge.LinalgInv(x, s)
	if err := bridge.Eval(inv); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	out := make([]float32, 4)
	copy(out, unsafe.Slice((*float32)(inv.DataPtr()), 4))
	for i, v := range out {
		if math.Abs(float64(v)-float64(data[i])) > 0.001 {
			t.Errorf("inv[%d]: expected %f, got %f", i, data[i], v)
		}
	}
	t.Logf("Inv(I) = %v", out)
	inv.Free()
	x.Free()
}

func TestLinalgSolve(t *testing.T) {
	s := bridge.DefaultCPUStream() // linalg ops require CPU stream

	// Solve Ax = b where A = [[2,1],[1,3]], b = [5,7] → x = [1.6, 1.8]
	aData := []float32{2, 1, 1, 3}
	bData := []float32{5, 7}
	a := bridge.NewArrayFromData(unsafe.Pointer(&aData[0]), []int{2, 2}, bridge.DTypeFloat32)
	bArr := bridge.NewArrayFromData(unsafe.Pointer(&bData[0]), []int{2}, bridge.DTypeFloat32)

	result := bridge.LinalgSolve(a, bArr, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	out := make([]float32, 2)
	copy(out, unsafe.Slice((*float32)(result.DataPtr()), 2))
	if math.Abs(float64(out[0])-1.6) > 0.01 || math.Abs(float64(out[1])-1.8) > 0.01 {
		t.Errorf("solve: expected [1.6, 1.8], got %v", out)
	}
	t.Logf("Solve: %v", out)
	result.Free()
	a.Free()
	bArr.Free()
}

func TestLinalgQR(t *testing.T) {
	s := bridge.DefaultCPUStream() // linalg ops require CPU stream

	data := []float32{1, 2, 3, 4}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2, 2}, bridge.DTypeFloat32)

	q, r := bridge.LinalgQR(x, s)
	if err := bridge.Eval(q, r); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	t.Logf("QR: Q shape=%v, R shape=%v", q.Shape(), r.Shape())
	if q.NDim() != 2 || r.NDim() != 2 {
		t.Errorf("QR: expected 2D results, got Q=%dD, R=%dD", q.NDim(), r.NDim())
	}
	q.Free()
	r.Free()
	x.Free()
}

func TestLinalgNorm(t *testing.T) {
	s := bridge.DefaultGPUStream() // norm works on GPU

	data := []float32{3, 4}
	x := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2}, bridge.DTypeFloat32)

	// L2 norm of [3,4] = 5.
	result := bridge.LinalgNormL2(x, []int{0}, false, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	val := *(*float32)(result.DataPtr())
	if math.Abs(float64(val)-5.0) > 0.01 {
		t.Errorf("norm: expected 5.0, got %f", val)
	}
	t.Logf("L2 norm([3,4]) = %f", val)
	result.Free()
	x.Free()
}

// ===========================================================================
// SafeTensors IO Tests
// ===========================================================================

func TestSafeTensorsIO(t *testing.T) {
	s := bridge.DefaultCPUStream() // IO ops require CPU stream

	// Create test data.
	data := []float32{1, 2, 3, 4, 5, 6}
	arr := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{2, 3}, bridge.DTypeFloat32)
	if err := bridge.Eval(arr); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	// Build maps.
	arrays := bridge.NewMapStringToArray()
	arrays.Insert("weight", arr)

	metadata := bridge.NewMapStringToString()
	metadata.Insert("format", "pt")

	// Save.
	tmpPath := t.TempDir() + "/test.safetensors"
	if err := bridge.SaveSafeTensors(tmpPath, arrays, metadata); err != nil {
		t.Fatalf("SaveSafeTensors failed: %v", err)
	}

	// Load.
	loadedArrays, loadedMeta, err := bridge.LoadSafeTensors(tmpPath, s)
	if err != nil {
		t.Fatalf("LoadSafeTensors failed: %v", err)
	}

	// Verify arrays.
	w := loadedArrays.Get("weight")
	if w == nil {
		t.Fatal("loaded 'weight' is nil")
	}
	if err := bridge.Eval(w); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	shape := w.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("loaded weight shape: expected [2,3], got %v", shape)
	}
	out := make([]float32, 6)
	copy(out, unsafe.Slice((*float32)(w.DataPtr()), 6))
	for i, v := range out {
		if v != data[i] {
			t.Errorf("loaded weight[%d]: expected %f, got %f", i, data[i], v)
		}
	}

	// Verify metadata.
	format, ok := loadedMeta.Get("format")
	if !ok || format != "pt" {
		t.Errorf("metadata 'format': expected 'pt', got '%s' (ok=%v)", format, ok)
	}

	t.Logf("SafeTensors roundtrip OK: shape=%v, metadata format=%s", shape, format)

	arr.Free()
	w.Free()
	arrays.Free()
	metadata.Free()
	loadedArrays.Free()
	loadedMeta.Free()
}

// ===========================================================================
// Advanced RNG Tests
// ===========================================================================

func TestRandomBernoulli(t *testing.T) {
	s := bridge.DefaultGPUStream()

	pData := []float32{0.5}
	p := bridge.NewArrayFromData(unsafe.Pointer(&pData[0]), []int{1}, bridge.DTypeFloat32)
	key := bridge.RandomKey(42)

	result := bridge.RandomBernoulli(p, []int{100}, key, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	if result.Size() != 100 {
		t.Errorf("bernoulli size: expected 100, got %d", result.Size())
	}
	t.Logf("RandomBernoulli: shape=%v", result.Shape())
	result.Free()
	p.Free()
	key.Free()
}

func TestRandomRandInt(t *testing.T) {
	s := bridge.DefaultGPUStream()

	low := bridge.NewArrayScalarInt32(0)
	high := bridge.NewArrayScalarInt32(10)
	key := bridge.RandomKey(42)

	result := bridge.RandomRandInt(low, high, []int{5}, bridge.DTypeInt32, key, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	if result.Size() != 5 {
		t.Errorf("randint size: expected 5, got %d", result.Size())
	}
	out := make([]int32, 5)
	copy(out, unsafe.Slice((*int32)(result.DataPtr()), 5))
	for i, v := range out {
		if v < 0 || v >= 10 {
			t.Errorf("randint[%d]: %d not in [0, 10)", i, v)
		}
	}
	t.Logf("RandomRandInt: %v", out)
	result.Free()
	low.Free()
	high.Free()
	key.Free()
}

func TestRandomSeed(t *testing.T) {
	// Just verify it doesn't panic.
	bridge.RandomSeed(12345)
	t.Log("RandomSeed(12345) OK")
}

// ===========================================================================
// Transpose Convolution Tests
// ===========================================================================

func TestConvTranspose1d(t *testing.T) {
	s := bridge.DefaultGPUStream()

	// input: [batch=1, length=4, channels_in=1]
	inputData := []float32{1, 2, 3, 4}
	input := bridge.NewArrayFromData(unsafe.Pointer(&inputData[0]), []int{1, 4, 1}, bridge.DTypeFloat32)

	// weight: [channels_out=1, kernel=2, channels_in=1]
	weightData := []float32{1, 1}
	weight := bridge.NewArrayFromData(unsafe.Pointer(&weightData[0]), []int{1, 2, 1}, bridge.DTypeFloat32)

	result := bridge.ConvTranspose1d(input, weight, 1, 0, 1, 0, 1, s)
	if err := bridge.Eval(result); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}
	shape := result.Shape()
	t.Logf("ConvTranspose1d: shape=%v", shape)
	if shape[0] != 1 { // batch
		t.Errorf("expected batch=1, got %d", shape[0])
	}
	result.Free()
	input.Free()
	weight.Free()
}
