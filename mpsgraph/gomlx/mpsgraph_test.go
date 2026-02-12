// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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
