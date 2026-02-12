// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestBackendCreation tests that the backend can be created.
func TestBackendCreation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if backend.Name() != "mpsgraph" {
		t.Errorf("Name() = %q, want %q", backend.Name(), "mpsgraph")
	}
	if backend.NumDevices() != 1 {
		t.Errorf("NumDevices() = %d, want 1", backend.NumDevices())
	}
	t.Logf("Backend: %s — %s", backend.Name(), backend.Description())
}

// TestBufferOperations tests buffer creation, read-back, and finalize.
func TestBufferOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	buf, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	gotShape, err := backend.BufferShape(buf)
	if err != nil {
		t.Fatalf("BufferShape() failed: %+v", err)
	}
	if !gotShape.Equal(shape) {
		t.Errorf("BufferShape() = %v, want %v", gotShape, shape)
	}

	outputData := make([]float32, 6)
	if err := backend.BufferToFlatData(buf, outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}

	if err := backend.BufferFinalize(buf); err != nil {
		t.Fatalf("BufferFinalize() failed: %+v", err)
	}
}

// TestSharedBuffer tests shared buffer (Go-managed memory) functionality.
func TestSharedBuffer(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if !backend.HasSharedBuffers() {
		t.Fatal("HasSharedBuffers() = false, want true")
	}

	shape := shapes.Make(dtypes.Float32, 4)
	buf, flat, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer() failed: %+v", err)
	}

	flatData := flat.([]float32)
	for i := range flatData {
		flatData[i] = float32(i + 1)
	}

	gotFlat, err := backend.BufferData(buf)
	if err != nil {
		t.Fatalf("BufferData() failed: %+v", err)
	}
	gotData := gotFlat.([]float32)
	for i := range gotData {
		if gotData[i] != float32(i+1) {
			t.Errorf("gotData[%d] = %f, want %f", i, gotData[i], float32(i+1))
		}
	}
	_ = backend.BufferFinalize(buf)
}

// execUnaryOp is a test helper that builds, compiles, and executes a unary op.
func execUnaryOp(t *testing.T, backend backends.Backend, opFn func(backends.Function, backends.Value) (backends.Value, error),
	input []float32) []float32 {
	t.Helper()

	shape := shapes.Make(dtypes.Float32, len(input))
	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := opFn(mainFn, x)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, input, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(input))
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	return result
}

// execBinaryOp is a test helper that builds, compiles, and executes a binary op.
func execBinaryOp(t *testing.T, backend backends.Backend,
	opFn func(backends.Function, backends.Value, backends.Value) (backends.Value, error),
	shape shapes.Shape, lhsData, rhsData []float32) []float32 {
	t.Helper()

	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	z, err := opFn(mainFn, x, y)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{z}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %+v", err)
	}
	yBuf, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(y) failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(lhsData))
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	return result
}

func assertClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

// TestAddOperation tests the full pipeline: parameter → add → compile → execute → read.
func TestAddOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	result := execBinaryOp(t, backend,
		func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Add(x, y) },
		shape,
		[]float32{1, 2, 3, 4, 5, 6},
		[]float32{10, 20, 30, 40, 50, 60},
	)
	assertClose(t, result, []float32{11, 22, 33, 44, 55, 66}, 1e-5)
}

// TestUnaryOperations tests several unary math ops.
func TestUnaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	tests := []struct {
		name     string
		opFn     func(backends.Function, backends.Value) (backends.Value, error)
		input    []float32
		expected []float32
		tol      float64
	}{
		{"Abs", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Abs(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, 2, 3, 4}, 1e-5},
		{"Neg", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Neg(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, -2, 3, -4}, 1e-5},
		{"Sqrt", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Sqrt(x) },
			[]float32{1, 4, 9, 16}, []float32{1, 2, 3, 4}, 1e-5},
		{"Exp", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Exp(x) },
			[]float32{0, 1}, []float32{1, float32(math.E)}, 1e-5},
		{"Log", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Log(x) },
			[]float32{1, float32(math.E)}, []float32{0, 1}, 1e-4},
		{"Tanh", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Tanh(x) },
			[]float32{0, 1}, []float32{0, float32(math.Tanh(1))}, 1e-5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := execUnaryOp(t, backend, tc.opFn, tc.input)
			assertClose(t, got, tc.expected, tc.tol)
		})
	}
}

// TestBinaryOperations tests binary math ops.
func TestBinaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhs := []float32{10, 20, 30, 40}
	rhs := []float32{3, 5, 10, 8}

	t.Run("Sub", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Sub(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{7, 15, 20, 32}, 1e-5)
	})

	t.Run("Mul", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Mul(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{30, 100, 300, 320}, 1e-5)
	})

	t.Run("Div", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Div(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10.0 / 3, 4, 3, 5}, 1e-4)
	})

	t.Run("Max", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Max(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10, 20, 30, 40}, 1e-5)
	})

	t.Run("Min", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Min(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{3, 5, 10, 8}, 1e-5)
	})
}

// TestReshape tests the Reshape operation.
func TestReshape(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Reshape(x, 3, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	outShapes := exec.Outputs()
	if len(outShapes) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outShapes))
	}
	expectedOutShape := shapes.Make(dtypes.Float32, 3, 2)
	if !outShapes[0].Equal(expectedOutShape) {
		t.Errorf("Output shape = %v, want %v", outShapes[0], expectedOutShape)
	}

	xData := []float32{1, 2, 3, 4, 5, 6}
	xBuf, err := backend.BufferFromFlatData(0, xData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Reshape doesn't change data, just layout.
	assertClose(t, result, xData, 0)
}

// TestConstant tests that Constant values are compiled into the graph correctly.
func TestConstant(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_constant")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Create a constant and add it to the parameter.
	c, err := mainFn.Constant([]float32{10, 20, 30}, 3)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}

	y, err := mainFn.Add(x, c)
	if err != nil {
		t.Fatalf("Add() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{11, 22, 33}, 1e-5)
}

// TestReduceSum tests the ReduceSum operation.
func TestReduceSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_sum")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Sum over axis 1 → shape [2].
	y, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 2)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{6, 15}, 1e-5)
}

// TestDotGeneral tests the DotGeneral (batched matmul) operation.
func TestDotGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("SimpleMatMul", func(t *testing.T) {
		// [2,3] x [3,2] → [2,2]
		builder := backend.Builder("test_dot_simple")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		// Standard matmul: contract axis 1 of lhs with axis 0 of rhs.
		y, err := mainFn.DotGeneral(lhs, []int{1}, nil, rhs, []int{0}, nil)
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// lhs = [[1,2,3],[4,5,6]], rhs = [[1,2],[3,4],[5,6]]
		// result = [[22,28],[49,64]]
		lhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, rhsShape)

		outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 4)
		if err := backend.BufferToFlatData(outputs[0], result); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64}, 1e-4)
	})

	t.Run("BatchedMatMul", func(t *testing.T) {
		// [2,2,3] x [2,3,2] → [2,2,2] with batch axis 0.
		builder := backend.Builder("test_dot_batched")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 2, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		y, err := mainFn.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0})
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// Batch 0: [[1,2,3],[4,5,6]] x [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
		// Batch 1: [[7,8,9],[10,11,12]] x [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
		lhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		rhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		lhsBuf, _ := backend.BufferFromFlatData(0, lhsData, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, rhsData, rhsShape)

		outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 8)
		if err := backend.BufferToFlatData(outputs[0], result); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64, 220, 244, 301, 334}, 1e-3)
	})
}

// TestGather tests the Gather operation (embedding lookup pattern).
func TestGather(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("EmbeddingLookup", func(t *testing.T) {
		// Operand: [4, 3] embedding table, indices: [2] → gather rows 1, 3 → [2, 3]
		builder := backend.Builder("test_gather_embed")
		mainFn := builder.Main()

		operandShape := shapes.Make(dtypes.Float32, 4, 3)
		indicesShape := shapes.Make(dtypes.Int32, 2, 1) // [2, 1] with indexVectorAxis=1

		operand, err := mainFn.Parameter("operand", operandShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		indices, err := mainFn.Parameter("indices", indicesShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		// XLA-style gather for embedding lookup:
		// indexVectorAxis=1, startIndexMap=[0], collapsedSliceAxes=[0],
		// offsetOutputAxes=[1], sliceSizes=[1, 3]
		result, err := mainFn.Gather(operand, indices,
			1,        // indexVectorAxis
			[]int{1}, // offsetOutputAxes
			[]int{0}, // collapsedSliceAxes
			[]int{0}, // startIndexMap
			[]int{1, 3}, // sliceSizes
			false,       // indicesAreSorted
		)
		if err != nil {
			t.Fatalf("Gather() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// Embedding table: [[10,11,12], [20,21,22], [30,31,32], [40,41,42]]
		// Indices: [1, 3] → select rows 1 and 3
		operandData := []float32{10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42}
		indicesData := []int32{1, 3}

		operandBuf, err := backend.BufferFromFlatData(0, operandData, operandShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(operand) failed: %+v", err)
		}
		indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(indices) failed: %+v", err)
		}

		outputs, err := exec.Execute([]backends.Buffer{operandBuf, indicesBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		got := make([]float32, 6)
		if err := backend.BufferToFlatData(outputs[0], got); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		// Expected: rows 1 and 3 → [20,21,22, 40,41,42]
		assertClose(t, got, []float32{20, 21, 22, 40, 41, 42}, 1e-5)
	})
}

// TestSlice tests the Slice operation.
func TestSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_slice")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3, 4)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Slice rows [1:3], cols [1:3] → [2, 2]
	y, err := mainFn.Slice(x, []int{1, 1}, []int{3, 3}, []int{1, 1})
	if err != nil {
		t.Fatalf("Slice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	xData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	xBuf, err := backend.BufferFromFlatData(0, xData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[6,7], [10,11]]
	assertClose(t, got, []float32{6, 7, 10, 11}, 1e-5)
}

// TestConcatenate tests the Concatenate operation.
func TestConcatenate(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_concat")
	mainFn := builder.Main()

	shapeA := shapes.Make(dtypes.Float32, 2, 2)
	shapeB := shapes.Make(dtypes.Float32, 2, 3)

	a, err := mainFn.Parameter("a", shapeA, nil)
	if err != nil {
		t.Fatalf("Parameter(a) failed: %+v", err)
	}
	b, err := mainFn.Parameter("b", shapeB, nil)
	if err != nil {
		t.Fatalf("Parameter(b) failed: %+v", err)
	}

	// Concatenate along axis 1: [2,2] + [2,3] → [2,5]
	y, err := mainFn.Concatenate(1, a, b)
	if err != nil {
		t.Fatalf("Concatenate() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	aBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapeA)
	bBuf, _ := backend.BufferFromFlatData(0, []float32{5, 6, 7, 8, 9, 10}, shapeB)

	outputs, err := exec.Execute([]backends.Buffer{aBuf, bBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 10)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[1,2,5,6,7], [3,4,8,9,10]]
	assertClose(t, got, []float32{1, 2, 5, 6, 7, 3, 4, 8, 9, 10}, 1e-5)
}

// TestIota tests the Iota operation.
func TestIota(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_iota")
	mainFn := builder.Main()

	// Iota with shape [2, 3] along axis 1 → [[0,1,2],[0,1,2]]
	outShape := shapes.Make(dtypes.Float32, 2, 3)
	y, err := mainFn.Iota(outShape, 1)
	if err != nil {
		t.Fatalf("Iota() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute(nil, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{0, 1, 2, 0, 1, 2}, 1e-5)
}

// TestBroadcastInDim tests the BroadcastInDim operation.
func TestBroadcastInDim(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_broadcast")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Broadcast [3] → [2, 3] by mapping axis 0 of input to axis 1 of output.
	outShape := shapes.Make(dtypes.Float32, 2, 3)
	y, err := mainFn.BroadcastInDim(x, outShape, []int{1})
	if err != nil {
		t.Fatalf("BroadcastInDim() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[10,20,30], [10,20,30]]
	assertClose(t, got, []float32{10, 20, 30, 10, 20, 30}, 1e-5)
}

// TestPad tests the Pad operation.
func TestPad(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_pad")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 2)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Pad value = 0
	padVal, err := mainFn.Constant([]float32{0}, 1)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}
	padValScalar, err := mainFn.Reshape(padVal)
	if err != nil {
		t.Fatalf("Reshape() failed: %+v", err)
	}

	// Pad: 1 before axis 0, 0 after; 0 before axis 1, 1 after → [3, 3]
	y, err := mainFn.Pad(x, padValScalar,
		backends.PadAxis{Start: 1, End: 0},
		backends.PadAxis{Start: 0, End: 1},
	)
	if err != nil {
		t.Fatalf("Pad() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 9)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[0,0,0], [1,2,0], [3,4,0]]
	assertClose(t, got, []float32{0, 0, 0, 1, 2, 0, 3, 4, 0}, 1e-5)
}

// TestArgMinMax tests the ArgMinMax operation.
func TestArgMinMax(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_argmax")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// ArgMax along axis 1 → shape [2], dtype Int32.
	y, err := mainFn.ArgMinMax(x, 1, dtypes.Int32, false)
	if err != nil {
		t.Fatalf("ArgMinMax() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 5, 3], [8, 2, 6]] → argmax on axis 1 → [1, 0]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 5, 3, 8, 2, 6}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]int32, 2)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	if got[0] != 1 || got[1] != 0 {
		t.Errorf("ArgMax got %v, want [1, 0]", got)
	}
}

// TestDynamicSlice tests the DynamicSlice operation.
func TestDynamicSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_slice")
	mainFn := builder.Main()

	// Input: [3, 4], slice a [2, 2] window starting at (1, 1).
	inShape := shapes.Make(dtypes.Float32, 3, 4)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Start indices as scalar Int32 parameters.
	startShape := shapes.Make(dtypes.Int32)
	start0, err := mainFn.Parameter("start0", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start0) failed: %+v", err)
	}
	start1, err := mainFn.Parameter("start1", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start1) failed: %+v", err)
	}

	y, err := mainFn.DynamicSlice(x, []backends.Value{start0, start1}, []int{2, 2})
	if err != nil {
		t.Fatalf("DynamicSlice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	// Slice at (1,1) with size (2,2) → [[6,7], [10,11]]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %+v", err)
	}
	s0Buf, err := backend.BufferFromFlatData(0, []int32{1}, startShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(s0) failed: %+v", err)
	}
	s1Buf, err := backend.BufferFromFlatData(0, []int32{1}, startShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(s1) failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf, s0Buf, s1Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{6, 7, 10, 11}, 1e-5)
}

// TestDynamicUpdateSlice tests the DynamicUpdateSlice operation.
func TestDynamicUpdateSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice")
	mainFn := builder.Main()

	// Input: [3, 4], update: [2, 2] at position (1, 1).
	inShape := shapes.Make(dtypes.Float32, 3, 4)
	updateShape := shapes.Make(dtypes.Float32, 2, 2)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter(x) failed: %+v", err)
	}
	upd, err := mainFn.Parameter("upd", updateShape, nil)
	if err != nil {
		t.Fatalf("Parameter(upd) failed: %+v", err)
	}

	startShape := shapes.Make(dtypes.Int32)
	start0, err := mainFn.Parameter("start0", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start0) failed: %+v", err)
	}
	start1, err := mainFn.Parameter("start1", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start1) failed: %+v", err)
	}

	y, err := mainFn.DynamicUpdateSlice(x, upd, []backends.Value{start0, start1})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	// Update: [[100,200], [300,400]] at (1,1)
	// Expected: [[1,2,3,4], [5,100,200,8], [9,300,400,12]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, inShape)
	updBuf, _ := backend.BufferFromFlatData(0, []float32{100, 200, 300, 400}, updateShape)
	s0Buf, _ := backend.BufferFromFlatData(0, []int32{1}, startShape)
	s1Buf, _ := backend.BufferFromFlatData(0, []int32{1}, startShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf, updBuf, s0Buf, s1Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 12)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{1, 2, 3, 4, 5, 100, 200, 8, 9, 300, 400, 12}, 1e-5)
}

// TestFusedSoftmax tests the FusedSoftmax operation.
func TestFusedSoftmax(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_softmax")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Softmax along axis 1.
	y, err := mainFn.FusedSoftmax(x, 1)
	if err != nil {
		t.Fatalf("FusedSoftmax() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 2, 3], [1, 1, 1]]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 1, 1, 1}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Row 0: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
	// Row 1: softmax([1,1,1]) = [0.3333, 0.3333, 0.3333]
	expected := []float32{
		float32(math.Exp(1) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(2) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(3) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
	}
	assertClose(t, got, expected, 1e-4)
}

// TestScatterSum tests the ScatterSum operation (embedding gradient pattern).
func TestScatterSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	// Operand: [4, 2] (e.g., gradient accumulator for 4-row embedding).
	// Indices: [3, 1] (scatter to rows 0, 2, 0 — row 0 gets two updates summed).
	// Updates: [3, 2] (values to scatter-add).
	builder := backend.Builder("test_scatter_sum")
	mainFn := builder.Main()

	operandShape := shapes.Make(dtypes.Float32, 4, 2)
	indicesShape := shapes.Make(dtypes.Int32, 3, 1)
	updatesShape := shapes.Make(dtypes.Float32, 3, 2)

	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter(operand) failed: %+v", err)
	}
	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter(indices) failed: %+v", err)
	}
	updates, err := mainFn.Parameter("updates", updatesShape, nil)
	if err != nil {
		t.Fatalf("Parameter(updates) failed: %+v", err)
	}

	// ScatterSum: scatter updates into operand along axis 0.
	// indexVectorAxis=1, updateWindowAxes=[1], insertedWindowAxes=[0],
	// scatterAxesToOperandAxes=[0]
	result, err := mainFn.ScatterSum(operand, indices, updates,
		1,        // indexVectorAxis
		[]int{1}, // updateWindowAxes
		[]int{0}, // insertedWindowAxes
		[]int{0}, // scatterAxesToOperandAxes
		false, false,
	)
	if err != nil {
		t.Fatalf("ScatterSum() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Operand (all zeros): [[0,0], [0,0], [0,0], [0,0]]
	// Indices: [0, 2, 0] → scatter-add to rows 0, 2, 0
	// Updates: [[1,2], [3,4], [5,6]]
	// Expected: row 0 = [0,0] + [1,2] + [5,6] = [6,8], row 2 = [0,0] + [3,4] = [3,4]
	operandBuf, _ := backend.BufferFromFlatData(0, []float32{0, 0, 0, 0, 0, 0, 0, 0}, operandShape)
	indicesBuf, _ := backend.BufferFromFlatData(0, []int32{0, 2, 0}, indicesShape)
	updatesBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, updatesShape)

	outputs, err := exec.Execute([]backends.Buffer{operandBuf, indicesBuf, updatesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 8)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[6,8], [0,0], [3,4], [0,0]]
	assertClose(t, got, []float32{6, 8, 0, 0, 3, 4, 0, 0}, 1e-5)
}

// TestReverse tests the Reverse operation.
func TestReverse(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reverse")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Reverse along axis 1.
	y, err := mainFn.Reverse(x, 1)
	if err != nil {
		t.Fatalf("Reverse() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[3,2,1], [6,5,4]]
	assertClose(t, got, []float32{3, 2, 1, 6, 5, 4}, 1e-5)
}

// TestReduceProduct tests the ReduceProduct operation.
func TestReduceProduct(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_product")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Product over axis 1 → shape [2].
	y, err := mainFn.ReduceProduct(x, 1)
	if err != nil {
		t.Fatalf("ReduceProduct() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 2)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [1*2*3=6, 4*5*6=120]
	assertClose(t, got, []float32{6, 120}, 1e-5)
}

// TestComparisonOps tests comparison operations.
func TestComparisonOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhsData := []float32{1, 2, 3, 4}
	rhsData := []float32{2, 2, 2, 2}

	tests := []struct {
		name     string
		opFn     func(backends.Function, backends.Value, backends.Value) (backends.Value, error)
		expected []bool
	}{
		{"Equal", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Equal(x, y) },
			[]bool{false, true, false, false}},
		{"NotEqual", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.NotEqual(x, y) },
			[]bool{true, false, true, true}},
		{"GreaterThan", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.GreaterThan(x, y) },
			[]bool{false, false, true, true}},
		{"LessThan", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.LessThan(x, y) },
			[]bool{true, false, false, false}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_cmp_" + tc.name)
			mainFn := builder.Main()

			x, _ := mainFn.Parameter("x", shape, nil)
			y, _ := mainFn.Parameter("y", shape, nil)
			z, err := tc.opFn(mainFn, x, y)
			if err != nil {
				t.Fatalf("opFn() failed: %+v", err)
			}
			if err := mainFn.Return([]backends.Value{z}, nil); err != nil {
				t.Fatalf("Return() failed: %+v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile() failed: %+v", err)
			}
			defer exec.Finalize()

			xBuf, _ := backend.BufferFromFlatData(0, lhsData, shape)
			yBuf, _ := backend.BufferFromFlatData(0, rhsData, shape)

			outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %+v", err)
			}

			got := make([]bool, 4)
			if err := backend.BufferToFlatData(outputs[0], got); err != nil {
				t.Fatalf("BufferToFlatData() failed: %+v", err)
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Errorf("[%d] = %v, want %v", i, got[i], tc.expected[i])
				}
			}
		})
	}
}

// TestWhere tests the Where (select) operation.
func TestWhere(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_where")
	mainFn := builder.Main()

	dataShape := shapes.Make(dtypes.Float32, 4)
	condShape := shapes.Make(dtypes.Bool, 4)

	cond, _ := mainFn.Parameter("cond", condShape, nil)
	onTrue, _ := mainFn.Parameter("onTrue", dataShape, nil)
	onFalse, _ := mainFn.Parameter("onFalse", dataShape, nil)

	result, err := mainFn.Where(cond, onTrue, onFalse)
	if err != nil {
		t.Fatalf("Where() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	condBuf, _ := backend.BufferFromFlatData(0, []bool{true, false, true, false}, condShape)
	trueBuf, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40}, dataShape)
	falseBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, dataShape)

	outputs, err := exec.Execute([]backends.Buffer{condBuf, trueBuf, falseBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{10, 2, 30, 4}, 1e-5)
}

// TestConvertDType tests dtype conversion.
func TestConvertDType(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_convert")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Convert Float32 → Int32 (truncation).
	y, err := mainFn.ConvertDType(x, dtypes.Int32)
	if err != nil {
		t.Fatalf("ConvertDType() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, _ := backend.BufferFromFlatData(0, []float32{1.7, 2.3, -0.9}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]int32, 3)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Float32 → Int32 truncates toward zero.
	if got[0] != 1 || got[1] != 2 || got[2] != 0 {
		t.Errorf("ConvertDType got %v, want [1, 2, 0]", got)
	}
}

// TestTranspose tests the Transpose operation.
func TestTranspose(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transpose")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Transpose [2,3] → [3,2]
	y, err := mainFn.Transpose(x, 1, 0)
	if err != nil {
		t.Fatalf("Transpose() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]] (row-major: [1,2,3,4,5,6])
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Transposed: [[1,4], [2,5], [3,6]] (row-major: [1,4,2,5,3,6])
	assertClose(t, got, []float32{1, 4, 2, 5, 3, 6}, 1e-5)
}

// TestMultipleOutputs tests returning multiple values from a graph.
func TestMultipleOutputs(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_multi_output")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 3)
	x, _ := mainFn.Parameter("x", shape, nil)
	y, _ := mainFn.Parameter("y", shape, nil)

	sum, _ := mainFn.Add(x, y)
	diff, _ := mainFn.Sub(x, y)

	if err := mainFn.Return([]backends.Value{sum, diff}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	if len(exec.Outputs()) != 2 {
		t.Fatalf("Expected 2 outputs, got %d", len(exec.Outputs()))
	}

	xBuf, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30}, shape)
	yBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	gotSum := make([]float32, 3)
	gotDiff := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], gotSum); err != nil {
		t.Fatalf("BufferToFlatData(sum) failed: %+v", err)
	}
	if err := backend.BufferToFlatData(outputs[1], gotDiff); err != nil {
		t.Fatalf("BufferToFlatData(diff) failed: %+v", err)
	}

	assertClose(t, gotSum, []float32{11, 22, 33}, 1e-5)
	assertClose(t, gotDiff, []float32{9, 18, 27}, 1e-5)
}

// TestTransformerBlock tests a simplified transformer-style computation:
// attention_scores = softmax(Q @ K^T / sqrt(d_k)) @ V
func TestTransformerBlock(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transformer")
	mainFn := builder.Main()

	// Q, K, V: [2, 3] (seq_len=2, d_k=3)
	qkvShape := shapes.Make(dtypes.Float32, 2, 3)
	Q, _ := mainFn.Parameter("Q", qkvShape, nil)
	K, _ := mainFn.Parameter("K", qkvShape, nil)
	V, _ := mainFn.Parameter("V", qkvShape, nil)

	// scores = Q @ K^T → [2, 2]
	scores, err := mainFn.DotGeneral(Q, []int{1}, nil, K, []int{1}, nil)
	if err != nil {
		t.Fatalf("DotGeneral(Q, K^T) failed: %+v", err)
	}

	// Scale by 1/sqrt(d_k) = 1/sqrt(3)
	scaleFactor := float32(1.0 / math.Sqrt(3.0))
	scaleConst, _ := mainFn.Constant([]float32{scaleFactor}, 1)
	scaleShape := shapes.Make(dtypes.Float32, 2, 2)
	scaleBroadcast, _ := mainFn.BroadcastInDim(scaleConst, scaleShape, []int{1})
	scores, _ = mainFn.Mul(scores, scaleBroadcast)

	// weights = softmax(scores, axis=1)
	weights, err := mainFn.FusedSoftmax(scores, 1)
	if err != nil {
		t.Fatalf("FusedSoftmax() failed: %+v", err)
	}

	// output = weights @ V → [2, 3]
	output, err := mainFn.DotGeneral(weights, []int{1}, nil, V, []int{0}, nil)
	if err != nil {
		t.Fatalf("DotGeneral(weights, V) failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Simple test data.
	qData := []float32{1, 0, 0, 0, 1, 0} // Q = identity-like rows
	kData := []float32{1, 0, 0, 0, 1, 0}
	vData := []float32{10, 20, 30, 40, 50, 60}

	qBuf, _ := backend.BufferFromFlatData(0, qData, qkvShape)
	kBuf, _ := backend.BufferFromFlatData(0, kData, qkvShape)
	vBuf, _ := backend.BufferFromFlatData(0, vData, qkvShape)

	outputs, err := exec.Execute([]backends.Buffer{qBuf, kBuf, vBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Verify output is a valid weighted combination of V rows.
	// Row 0 should weight V[0] more (since Q[0] dot K[0] = 1 but Q[0] dot K[1] = 0).
	// Row 1 should weight V[1] more (since Q[1] dot K[1] = 1 but Q[1] dot K[0] = 0).
	// After softmax with scale 1/sqrt(3):
	//   score[0] = [1/sqrt(3), 0] → softmax → [exp(1/sqrt(3))/(exp(1/sqrt(3))+1), 1/(exp(1/sqrt(3))+1)]
	s := math.Exp(1.0 / math.Sqrt(3.0))
	w0 := float32(s / (s + 1.0))
	w1 := float32(1.0 / (s + 1.0))

	// Row 0 of output = w0 * V[0] + w1 * V[1]
	expectedRow0 := []float32{w0*10 + w1*40, w0*20 + w1*50, w0*30 + w1*60}
	// Row 1 of output = w1 * V[0] + w0 * V[1]
	expectedRow1 := []float32{w1*10 + w0*40, w1*20 + w0*50, w1*30 + w0*60}
	expected := append(expectedRow0, expectedRow1...)

	assertClose(t, got, expected, 1e-3)
}

// TestChainedOps tests a multi-op graph: y = sigmoid(x * w + b).
func TestChainedOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_chained")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Constants: w and b.
	w, err := mainFn.Constant([]float32{0.5, -0.5, 1.0, -1.0}, 4)
	if err != nil {
		t.Fatalf("Constant(w) failed: %+v", err)
	}
	b, err := mainFn.Constant([]float32{0.1, 0.2, -0.1, 0.0}, 4)
	if err != nil {
		t.Fatalf("Constant(b) failed: %+v", err)
	}

	// y = sigmoid(x * w + b)
	xw, err := mainFn.Mul(x, w)
	if err != nil {
		t.Fatalf("Mul() failed: %+v", err)
	}
	xwb, err := mainFn.Add(xw, b)
	if err != nil {
		t.Fatalf("Add() failed: %+v", err)
	}
	y, err := mainFn.Logistic(xwb)
	if err != nil {
		t.Fatalf("Logistic() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xData := []float32{2.0, 3.0, -1.0, 0.5}
	xBuf, err := backend.BufferFromFlatData(0, xData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Compute expected: sigmoid(x*w + b) for each element.
	sigmoid := func(x float64) float32 { return float32(1.0 / (1.0 + math.Exp(-x))) }
	expected := []float32{
		sigmoid(2.0*0.5 + 0.1),   // sigmoid(1.1)
		sigmoid(3.0*-0.5 + 0.2),  // sigmoid(-1.3)
		sigmoid(-1.0*1.0 + -0.1), // sigmoid(-1.1)
		sigmoid(0.5*-1.0 + 0.0),  // sigmoid(-0.5)
	}
	assertClose(t, result, expected, 1e-5)
}

// TestConvGeneral tests the ConvGeneral (2D convolution) operation.
func TestConvGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_conv")
	mainFn := builder.Main()

	// NCHW format: input [1, 1, 4, 4], kernel [1, 1, 3, 3]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	kernelShape := shapes.Make(dtypes.Float32, 1, 1, 3, 3)

	input, _ := mainFn.Parameter("input", inputShape, nil)
	kernel, _ := mainFn.Parameter("kernel", kernelShape, nil)

	// Standard 2D conv: NCHW layout, stride 1, no padding.
	axes := backends.ConvolveAxesConfig{
		InputBatch: 0, InputChannels: 1, InputSpatial: []int{2, 3},
		KernelOutputChannels: 0, KernelInputChannels: 1, KernelSpatial: []int{2, 3},
		OutputBatch: 0, OutputChannels: 1, OutputSpatial: []int{2, 3},
	}

	output, err := mainFn.ConvGeneral(input, kernel, axes,
		[]int{1, 1},       // strides
		[][2]int{{0, 0}, {0, 0}}, // paddings
		nil,               // inputDilations
		nil,               // kernelDilations
		1, 1,              // groups
	)
	if err != nil {
		t.Fatalf("ConvGeneral() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: 4x4 matrix of all 1s.
	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = 1
	}
	// Kernel: 3x3 matrix of all 1s → sum pooling.
	kernelData := make([]float32, 9)
	for i := range kernelData {
		kernelData[i] = 1
	}

	inputBuf, _ := backend.BufferFromFlatData(0, inputData, inputShape)
	kernelBuf, _ := backend.BufferFromFlatData(0, kernelData, kernelShape)

	outputs, err := exec.Execute([]backends.Buffer{inputBuf, kernelBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	// Output shape: [1, 1, 2, 2] (valid conv with 3x3 kernel on 4x4 input).
	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Each output element is sum of 3x3 window of all 1s = 9.
	assertClose(t, got, []float32{9, 9, 9, 9}, 1e-5)
}

// TestReduceWindow tests the ReduceWindow (pooling) operation.
func TestReduceWindow(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_pool")
	mainFn := builder.Main()

	// NCHW format: [1, 1, 4, 4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	input, _ := mainFn.Parameter("input", inputShape, nil)

	// Max pool with 2x2 window, stride 2 → [1, 1, 2, 2]
	output, err := mainFn.ReduceWindow(input,
		backends.ReduceOpMax,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,                // baseDilations
		nil,                // windowDilations
		[][2]int{{0, 0}, {0, 0}, {0, 0}, {0, 0}}, // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: 4x4 matrix
	// [[1,  2,  3,  4],
	//  [5,  6,  7,  8],
	//  [9,  10, 11, 12],
	//  [13, 14, 15, 16]]
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	inputBuf, _ := backend.BufferFromFlatData(0, inputData, inputShape)

	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Max pool: max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
	assertClose(t, got, []float32{6, 8, 14, 16}, 1e-5)
}

// ===========================================================================
// GoMLX High-Level Integration Tests
// These tests use GoMLX's graph-building API (graph.Node, graph.Exec) to verify
// the mpsgraph backend works correctly with GoMLX's full computation graph system.
// ===========================================================================

// newTestBackend creates a backend for GoMLX integration tests.
func newTestBackend(t *testing.T) backends.Backend {
	t.Helper()
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	t.Cleanup(func() { backend.Finalize() })
	return backend
}

// TestGoMLXAdd tests using GoMLX's graph API for a simple add operation.
func TestGoMLXAdd(t *testing.T) {
	backend := newTestBackend(t)
	result := graph.MustExecOnce(backend, func(x, y *graph.Node) *graph.Node {
		return graph.Add(x, y)
	}, []float32{1, 2, 3}, []float32{10, 20, 30})

	got := result.Value().([]float32)
	assertClose(t, got, []float32{11, 22, 33}, 1e-5)
}

// TestGoMLXMathChain tests a chain of GoMLX math operations.
func TestGoMLXMathChain(t *testing.T) {
	backend := newTestBackend(t)
	// Compute: exp(log(x) + 1)
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.Exp(graph.Add(graph.Log(x), graph.Ones(x.Graph(), x.Shape())))
	}, []float32{1, 2, 4})

	got := result.Value().([]float32)
	// exp(log(x) + 1) = exp(log(x)) * exp(1) = x * e
	e := float32(math.E)
	assertClose(t, got, []float32{1 * e, 2 * e, 4 * e}, 1e-4)
}

// TestGoMLXMatMul tests matrix multiplication using GoMLX's graph API.
func TestGoMLXMatMul(t *testing.T) {
	backend := newTestBackend(t)
	lhs := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	rhs := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 3, 2)

	result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
		return graph.Dot(a, b)
	}, lhs, rhs)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	// [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
	assertClose(t, got, []float32{22, 28, 49, 64}, 1e-4)
}

// TestGoMLXReduceAndBroadcast tests reduce + broadcast using GoMLX API.
func TestGoMLXReduceAndBroadcast(t *testing.T) {
	backend := newTestBackend(t)
	input := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	// Compute row means and subtract (manual centering).
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		rowSums := graph.ReduceSum(x, -1)                  // [2]
		rowMeans := graph.DivScalar(rowSums, 3.0)           // [2]
		rowMeansBroadcast := graph.ExpandDims(rowMeans, -1) // [2, 1]
		return graph.Sub(x, rowMeansBroadcast)              // [2, 3] - [2, 1] → [2, 3]
	}, input)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	// Row 0 mean = (1+2+3)/3 = 2 → [-1, 0, 1]
	// Row 1 mean = (4+5+6)/3 = 5 → [-1, 0, 1]
	assertClose(t, got, []float32{-1, 0, 1, -1, 0, 1}, 1e-4)
}

// TestGoMLXGatherAndSlice tests gather/embedding lookup using GoMLX API.
func TestGoMLXGatherAndSlice(t *testing.T) {
	backend := newTestBackend(t)
	table := tensors.FromFlatDataAndDimensions([]float32{
		10, 11, 12,
		20, 21, 22,
		30, 31, 32,
		40, 41, 42,
	}, 4, 3)
	// For graph.Gather, the last dim of indices is the index vector dimension.
	// Shape [2, 1] means 2 indices, each indexing 1 axis of params → embedding lookup.
	indices := tensors.FromFlatDataAndDimensions([]int32{1, 3}, 2, 1)

	result := graph.MustExecOnce(backend, func(tbl, idx *graph.Node) *graph.Node {
		return graph.Gather(tbl, idx)
	}, table, indices)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{20, 21, 22, 40, 41, 42}, 1e-5)
}

// TestGoMLXWhereAndCompare tests conditional logic using GoMLX API.
func TestGoMLXWhereAndCompare(t *testing.T) {
	backend := newTestBackend(t)
	// ReLU: max(0, x) = where(x > 0, x, 0)
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		zero := graph.ZerosLike(x)
		mask := graph.GreaterThan(x, zero)
		return graph.Where(mask, x, zero)
	}, []float32{-2, -1, 0, 1, 2})

	got := result.Value().([]float32)
	assertClose(t, got, []float32{0, 0, 0, 1, 2}, 1e-5)
}
