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
// Bridge-level tests
// ===========================================================================

func TestMetalAvailable(t *testing.T) {
	if !bridge.MetalIsAvailable() {
		t.Fatal("Metal GPU not available")
	}
}

func TestArrayCreateAndRead(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	arr := bridge.NewArrayFromData(nil, []int{2, 3}, bridge.DTypeFloat32)
	defer arr.Free()

	// Create from data.
	arr2 := bridge.NewArrayFromData(
		unsafeFloat32Ptr(data),
		[]int{2, 3},
		bridge.DTypeFloat32,
	)
	defer arr2.Free()

	if err := bridge.Eval(arr2); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	shape := arr2.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("Expected shape [2, 3], got %v", shape)
	}
	if arr2.Size() != 6 {
		t.Fatalf("Expected size 6, got %d", arr2.Size())
	}
}

func TestBridgeArithmetic(t *testing.T) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	a := bridge.NewArrayFromData(unsafeFloat32Ptr([]float32{1, 2, 3}), []int{3}, bridge.DTypeFloat32)
	b := bridge.NewArrayFromData(unsafeFloat32Ptr([]float32{4, 5, 6}), []int{3}, bridge.DTypeFloat32)
	defer a.Free()
	defer b.Free()

	c := bridge.Add(a, b, s)
	defer c.Free()

	if err := bridge.Eval(c); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	result := readFloat32(c)
	expected := []float32{5, 7, 9}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("Add[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestBridgeMatMul(t *testing.T) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	// [2,3] @ [3,2] = [2,2]
	a := bridge.NewArrayFromData(unsafeFloat32Ptr([]float32{1, 2, 3, 4, 5, 6}), []int{2, 3}, bridge.DTypeFloat32)
	b := bridge.NewArrayFromData(unsafeFloat32Ptr([]float32{1, 4, 2, 5, 3, 6}), []int{3, 2}, bridge.DTypeFloat32)
	defer a.Free()
	defer b.Free()

	c := bridge.MatMul(a, b, s)
	defer c.Free()

	if err := bridge.Eval(c); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	shape := c.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Fatalf("Expected shape [2, 2], got %v", shape)
	}

	result := readFloat32(c)
	// [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
	// [4*1+5*2+6*3, 4*4+5*5+6*6] = [32, 77]
	expected := []float32{14, 32, 32, 77}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("MatMul[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestBridgeUnaryOps(t *testing.T) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	data := []float32{1, 4, 9, 16}
	x := bridge.NewArrayFromData(unsafeFloat32Ptr(data), []int{4}, bridge.DTypeFloat32)
	defer x.Free()

	sqrtX := bridge.Sqrt(x, s)
	defer sqrtX.Free()

	if err := bridge.Eval(sqrtX); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	result := readFloat32(sqrtX)
	expected := []float32{1, 2, 3, 4}
	for i, v := range result {
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Errorf("Sqrt[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestBridgeSoftmax(t *testing.T) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	data := []float32{1, 2, 3}
	x := bridge.NewArrayFromData(unsafeFloat32Ptr(data), []int{3}, bridge.DTypeFloat32)
	defer x.Free()

	sm := bridge.Softmax(x, 0, true, s)
	defer sm.Free()

	if err := bridge.Eval(sm); err != nil {
		t.Fatalf("Eval failed: %v", err)
	}

	result := readFloat32(sm)
	// Verify sum ≈ 1
	sum := float32(0)
	for _, v := range result {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum: expected 1.0, got %f", sum)
	}
}

// ===========================================================================
// Backend-level tests
// ===========================================================================

func TestBackendCreate(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer b.Finalize()

	if b.Name() != "mlx" {
		t.Errorf("Expected name 'mlx', got %q", b.Name())
	}
	if b.NumDevices() != 1 {
		t.Errorf("Expected 1 device, got %d", b.NumDevices())
	}
}

func TestBackendBuffer(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer b.Finalize()

	// Create buffer from flat data.
	flat := []float32{1, 2, 3, 4, 5, 6}
	shape := shapes.Make(dtypes.Float32, 2, 3)
	buf, err := b.BufferFromFlatData(0, flat, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %v", err)
	}
	defer b.BufferFinalize(buf)

	// Read back.
	result := make([]float32, 6)
	if err := b.BufferToFlatData(buf, result); err != nil {
		t.Fatalf("BufferToFlatData failed: %v", err)
	}

	for i, v := range result {
		if v != flat[i] {
			t.Errorf("Buffer[%d]: expected %f, got %f", i, flat[i], v)
		}
	}
}

func TestSimpleComputation(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer b.Finalize()

	// Build: z = x + y
	builder := b.Builder("test_add")
	main := builder.Main()

	xShape := shapes.Make(dtypes.Float32, 3)
	x, _ := main.Parameter("x", xShape, nil)
	y, _ := main.Parameter("y", xShape, nil)
	z, _ := main.(backends.Function).Add(x, y)
	main.(backends.Function).Return([]backends.Value{z}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	// Execute.
	xBuf, _ := b.BufferFromFlatData(0, []float32{1, 2, 3}, xShape)
	yBuf, _ := b.BufferFromFlatData(0, []float32{4, 5, 6}, xShape)

	results, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	result := make([]float32, 3)
	b.BufferToFlatData(results[0], result)

	expected := []float32{5, 7, 9}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("Add[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestMultipleOps(t *testing.T) {
	b, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer b.Finalize()

	// Build: z = (x * y) + x
	builder := b.Builder("test_mul_add")
	main := builder.Main()
	fn := main.(backends.Function)

	xShape := shapes.Make(dtypes.Float32, 4)
	x, _ := fn.Parameter("x", xShape, nil)
	y, _ := fn.Parameter("y", xShape, nil)
	xy, _ := fn.Mul(x, y)
	z, _ := fn.Add(xy, x)
	fn.Return([]backends.Value{z}, nil)

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Finalize()

	xBuf, _ := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, xShape)
	yBuf, _ := b.BufferFromFlatData(0, []float32{2, 3, 4, 5}, xShape)

	results, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	result := make([]float32, 4)
	b.BufferToFlatData(results[0], result)

	// x*y + x = [1*2+1, 2*3+2, 3*4+3, 4*5+4] = [3, 8, 15, 24]
	expected := []float32{3, 8, 15, 24}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("MulAdd[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

// ===========================================================================
// Helpers
// ===========================================================================

func unsafeFloat32Ptr(data []float32) unsafe.Pointer {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Pointer(&data[0])
}

func readFloat32(a *bridge.Array) []float32 {
	size := a.Size()
	result := make([]float32, size)
	ptr := (*float32)(a.DataPtr())
	for i := 0; i < size; i++ {
		result[i] = *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*4))
	}
	return result
}
