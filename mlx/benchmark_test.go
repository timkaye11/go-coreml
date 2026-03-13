// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// freeOutputs releases output buffers immediately to avoid GC/finalizer pressure.
func freeOutputs(outputs []backends.Buffer) {
	for _, buf := range outputs {
		if b, ok := buf.(*mlxBuffer); ok && b.array != nil {
			b.array.Free()
			b.array = nil
		}
	}
}

// BenchmarkMatMulCompilation benchmarks matrix multiplication compilation time.
func BenchmarkMatMulCompilation(b *testing.B) {
	sizes := []struct {
		name    string
		m, k, n int
	}{
		{"64x64x64", 64, 64, 64},
		{"128x128x128", 128, 128, 128},
		{"256x256x256", 256, 256, 256},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				builder := backend.Builder("bench_matmul")
				mainFn := builder.Main()

				lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
				rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

				lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for lhs failed: %v", err)
				}

				rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for rhs failed: %v", err)
				}

				result, err := mainFn.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{}, backends.DotGeneralConfig{})
				if err != nil {
					b.Fatalf("DotGeneral() failed: %v", err)
				}

				if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
					b.Fatalf("Return() failed: %v", err)
				}

				exec, err := builder.Compile()
				if err != nil {
					b.Fatalf("Compile() failed: %v", err)
				}
				exec.Finalize()
			}
		})
	}
}

// BenchmarkMatMulExecution benchmarks matrix multiplication execution time.
func BenchmarkMatMulExecution(b *testing.B) {
	sizes := []struct {
		name    string
		m, k, n int
	}{
		{"64x64x64", 64, 64, 64},
		{"128x128x128", 128, 128, 128},
		{"256x256x256", 256, 256, 256},
		{"512x512x512", 512, 512, 512},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			builder := backend.Builder("bench_matmul")
			mainFn := builder.Main()

			lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
			rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

			lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for lhs failed: %v", err)
			}

			rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for rhs failed: %v", err)
			}

			result, err := mainFn.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{}, backends.DotGeneralConfig{})
			if err != nil {
				b.Fatalf("DotGeneral() failed: %v", err)
			}

			if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
				b.Fatalf("Return() failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				b.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			lhsData := make([]float32, sz.m*sz.k)
			rhsData := make([]float32, sz.k*sz.n)
			for i := range lhsData {
				lhsData[i] = float32(i) * 0.01
			}
			for i := range rhsData {
				rhsData[i] = float32(i) * 0.01
			}

			lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for lhs failed: %v", err)
			}

			rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for rhs failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
				if err != nil {
					b.Fatalf("Execute() failed: %v", err)
				}
				freeOutputs(outputs)
			}
		})
	}
}

// BenchmarkUnaryOps benchmarks unary operations.
func BenchmarkUnaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_unary")
	mainFn := builder.Main()

	inputShape := shapes.Make(dtypes.Float32, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	absX, err := mainFn.Abs(x)
	if err != nil {
		b.Fatalf("Abs() failed: %v", err)
	}

	logX, err := mainFn.Log(absX)
	if err != nil {
		b.Fatalf("Log() failed: %v", err)
	}

	expX, err := mainFn.Exp(logX)
	if err != nil {
		b.Fatalf("Exp() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{expX}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	inputData := make([]float32, 1024)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.01
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		freeOutputs(outputs)
	}
}

// BenchmarkBinaryOps benchmarks binary operations.
func BenchmarkBinaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_binary")
	mainFn := builder.Main()

	inputShape := shapes.Make(dtypes.Float32, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() for x failed: %v", err)
	}

	y, err := mainFn.Parameter("y", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() for y failed: %v", err)
	}

	sum, err := mainFn.Add(x, y)
	if err != nil {
		b.Fatalf("Add() failed: %v", err)
	}

	diff, err := mainFn.Sub(x, y)
	if err != nil {
		b.Fatalf("Sub() failed: %v", err)
	}

	prod, err := mainFn.Mul(sum, diff)
	if err != nil {
		b.Fatalf("Mul() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{prod}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	xData := make([]float32, 1024)
	yData := make([]float32, 1024)
	for i := range xData {
		xData[i] = float32(i) * 0.01
		yData[i] = float32(1024-i) * 0.01
	}

	xBuf, err := backend.BufferFromFlatData(0, xData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() for x failed: %v", err)
	}

	yBuf, err := backend.BufferFromFlatData(0, yData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() for y failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		freeOutputs(outputs)
	}
}

// BenchmarkReduceOps benchmarks reduce operations.
func BenchmarkReduceOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_reduce")
	mainFn := builder.Main()

	inputShape := shapes.Make(dtypes.Float32, 1024, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	reduced, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		b.Fatalf("ReduceSum() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{reduced}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	inputData := make([]float32, 1024*1024)
	for i := range inputData {
		inputData[i] = float32(i) * 0.0001
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		freeOutputs(outputs)
	}
}

// BenchmarkMLPForward benchmarks a 2-layer MLP forward pass (FusedDense + Softmax),
// representative of real finetuning workloads.
func BenchmarkMLPForward(b *testing.B) {
	sizes := []struct {
		name                   string
		batch, input, hidden, output int
	}{
		{"B32_256_512_128", 32, 256, 512, 128},
		{"B64_512_1024_256", 64, 512, 1024, 256},
		{"B128_768_2048_512", 128, 768, 2048, 512},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			builder := backend.Builder("bench_mlp")
			mainFn := builder.Main()
			fn := mainFn.(backends.Function)

			xShape := shapes.Make(dtypes.Float32, sz.batch, sz.input)
			w1Shape := shapes.Make(dtypes.Float32, sz.input, sz.hidden)
			b1Shape := shapes.Make(dtypes.Float32, sz.hidden)
			w2Shape := shapes.Make(dtypes.Float32, sz.hidden, sz.output)
			b2Shape := shapes.Make(dtypes.Float32, sz.output)

			xParam, _ := fn.Parameter("x", xShape, nil)
			w1Param, _ := fn.Parameter("w1", w1Shape, nil)
			b1Param, _ := fn.Parameter("b1", b1Shape, nil)
			w2Param, _ := fn.Parameter("w2", w2Shape, nil)
			b2Param, _ := fn.Parameter("b2", b2Shape, nil)

			// Layer 1: ReLU(x @ W1 + b1)
			h, _ := fn.FusedDense(xParam, w1Param, b1Param, backends.ActivationRelu)
			// Layer 2: softmax(h @ W2 + b2)
			out, _ := fn.FusedDense(h, w2Param, b2Param, backends.ActivationNone)
			out, _ = fn.FusedSoftmax(out, 1)

			fn.Return([]backends.Value{out}, nil)

			exec, err := builder.Compile()
			if err != nil {
				b.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers with random-ish data.
			makeBuf := func(shape shapes.Shape) backends.Buffer {
				data := make([]float32, shape.Size())
				for i := range data {
					data[i] = float32(i%1000) * 0.001
				}
				buf, err := backend.BufferFromFlatData(0, data, shape)
				if err != nil {
					b.Fatalf("BufferFromFlatData() failed: %v", err)
				}
				return buf
			}

			inputs := []backends.Buffer{
				makeBuf(xShape), makeBuf(w1Shape), makeBuf(b1Shape),
				makeBuf(w2Shape), makeBuf(b2Shape),
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute(inputs, nil, 0)
				if err != nil {
					b.Fatalf("Execute() failed: %v", err)
				}
				freeOutputs(outputs)
			}
		})
	}
}
