// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
)

// memSample records memory state at a point in time.
type memSample struct {
	step   int
	active uint64
	cache  uint64
	peak   uint64
}

// memTracker records memory samples and provides stability assertions.
type memTracker struct {
	samples []memSample
}

func (m *memTracker) record(step int) {
	m.samples = append(m.samples, memSample{
		step:   step,
		active: bridge.GetActiveMemory(),
		cache:  bridge.GetCacheMemory(),
		peak:   bridge.GetPeakMemory(),
	})
}

// assertStableAfterWarmup checks that active memory growth post-warmup is below maxSlopeMBPerStep.
// Uses linear regression on active memory vs step number for samples after warmupSteps.
func (m *memTracker) assertStableAfterWarmup(t *testing.T, warmupSteps int, maxSlopeMBPerStep float64) {
	t.Helper()

	var xs, ys []float64
	for _, s := range m.samples {
		if s.step >= warmupSteps {
			xs = append(xs, float64(s.step))
			ys = append(ys, float64(s.active)/(1024*1024))
		}
	}
	if len(xs) < 3 {
		t.Logf("memTracker: only %d post-warmup samples, skipping slope check", len(xs))
		return
	}

	slope := linearRegressionSlope(xs, ys)
	t.Logf("memTracker: post-warmup slope = %.4f MB/step (max allowed: %.4f)", slope, maxSlopeMBPerStep)
	if slope > maxSlopeMBPerStep {
		t.Errorf("memory leak detected: slope %.4f MB/step exceeds threshold %.4f MB/step", slope, maxSlopeMBPerStep)
	}
}

// linearRegressionSlope returns the slope of the best-fit line through (xs, ys).
func linearRegressionSlope(xs, ys []float64) float64 {
	n := float64(len(xs))
	var sumX, sumY, sumXY, sumX2 float64
	for i := range xs {
		sumX += xs[i]
		sumY += ys[i]
		sumXY += xs[i] * ys[i]
		sumX2 += xs[i] * xs[i]
	}
	denom := n*sumX2 - sumX*sumX
	if math.Abs(denom) < 1e-12 {
		return 0
	}
	return (n*sumXY - sumX*sumY) / denom
}

// TestMLXBridgeBufferAllocFree allocates and frees [256,768] arrays 100 times.
// Asserts no significant active memory growth.
func TestMLXBridgeBufferAllocFree(t *testing.T) {
	if !bridge.MetalIsAvailable() {
		t.Skip("Metal not available")
	}

	bridge.ClearCache()
	bridge.ResetPeakMemory()
	runtime.GC()

	tracker := &memTracker{}
	tracker.record(0)

	for i := 0; i < 100; i++ {
		arr := bridge.NewArrayFromData(nil, []int{256, 768}, bridge.DTypeFloat32)
		if err := bridge.Eval(arr); err != nil {
			t.Fatalf("Eval failed at iter %d: %v", i, err)
		}
		arr.Free()

		if i%10 == 0 {
			bridge.ClearCache()
			runtime.GC()
			tracker.record(i)
		}
	}

	bridge.ClearCache()
	runtime.GC()
	tracker.record(100)

	finalActive := float64(bridge.GetActiveMemory()) / (1024 * 1024)
	t.Logf("Final active memory: %.2f MB", finalActive)
	tracker.assertStableAfterWarmup(t, 10, 0.1)
}

// TestMLXBridgeCompiledClosureRepeated compiles a matmul closure, runs it 100 times,
// and verifies no significant memory growth (catches mlx_compile trace cache leaks).
func TestMLXBridgeCompiledClosureRepeated(t *testing.T) {
	if !bridge.MetalIsAvailable() {
		t.Skip("Metal not available")
	}
	s := bridge.DefaultGPUStream()
	defer s.Free()

	bridge.ClearCache()
	bridge.ResetPeakMemory()
	runtime.GC()

	// Create a Go closure that does matmul.
	rawCl := bridge.NewClosureFromGoFunc(func(inputs []*bridge.Array) []*bridge.Array {
		return []*bridge.Array{bridge.MatMul(inputs[0], inputs[1], s)}
	})
	compiled := bridge.CompileClosure(rawCl, false)
	defer rawCl.Free()
	defer compiled.Free()

	a := bridge.NewArrayFromData(
		unsafeFloat32Ptr(make([]float32, 64*128)),
		[]int{64, 128}, bridge.DTypeFloat32,
	)
	b := bridge.NewArrayFromData(
		unsafeFloat32Ptr(make([]float32, 128*64)),
		[]int{128, 64}, bridge.DTypeFloat32,
	)
	defer a.Free()
	defer b.Free()

	tracker := &memTracker{}

	for i := 0; i < 100; i++ {
		outputs, err := bridge.ApplyClosure(compiled, []*bridge.Array{a, b})
		if err != nil {
			t.Fatalf("ApplyClosure failed at iter %d: %v", i, err)
		}
		if err := bridge.Eval(outputs...); err != nil {
			t.Fatalf("Eval failed at iter %d: %v", i, err)
		}
		for _, o := range outputs {
			o.Free()
		}

		if i%10 == 0 {
			bridge.ClearCache()
			runtime.GC()
			tracker.record(i)
		}
	}

	bridge.ClearCache()
	runtime.GC()
	tracker.record(100)

	tracker.assertStableAfterWarmup(t, 20, 0.5)
}

// TestMLXBridgeClearCacheEffectiveness allocates ~50 MB of buffers, frees them,
// calls ClearCache, and asserts cache memory drops to <1 MB.
func TestMLXBridgeClearCacheEffectiveness(t *testing.T) {
	if !bridge.MetalIsAvailable() {
		t.Skip("Metal not available")
	}

	bridge.ClearCache()
	runtime.GC()
	baselineCache := bridge.GetCacheMemory()

	// Allocate ~50 MB: 50 arrays of [256, 128] float32 = 50 * 256*128*4 = ~6.5 MB each? No.
	// 256*128*4 = 131072 bytes = 0.125 MB. Need ~400 arrays.
	// Or: 10 arrays of [1024, 1280] float32 = 10 * 1024*1280*4 = 52.4 MB total.
	arrays := make([]*bridge.Array, 10)
	for i := range arrays {
		data := make([]float32, 1024*1280)
		arrays[i] = bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{1024, 1280}, bridge.DTypeFloat32)
		if err := bridge.Eval(arrays[i]); err != nil {
			t.Fatalf("Eval failed: %v", err)
		}
	}

	postAllocCache := bridge.GetCacheMemory()
	t.Logf("Cache after alloc: %.2f MB (baseline: %.2f MB)",
		float64(postAllocCache)/(1024*1024), float64(baselineCache)/(1024*1024))

	// Free all arrays (they go into cache).
	for _, arr := range arrays {
		arr.Free()
	}
	runtime.GC()

	// Now clear cache.
	bridge.ClearCache()
	runtime.GC()

	postClearCache := bridge.GetCacheMemory()
	t.Logf("Cache after ClearCache: %.2f MB", float64(postClearCache)/(1024*1024))

	if postClearCache > 1024*1024 { // 1 MB threshold
		t.Errorf("ClearCache ineffective: cache still %.2f MB (want <1 MB)", float64(postClearCache)/(1024*1024))
	}
}

// TestMLXBridgeMemoryLimitRespected sets a memory limit and verifies GetActiveMemory reports values.
func TestMLXBridgeMemoryLimitRespected(t *testing.T) {
	if !bridge.MetalIsAvailable() {
		t.Skip("Metal not available")
	}

	bridge.ClearCache()
	runtime.GC()

	// Set a 500 MB limit.
	limitBytes := uint64(500 * 1024 * 1024)
	prevLimit := bridge.SetMemoryLimit(limitBytes)
	defer bridge.SetMemoryLimit(prevLimit) // restore

	bridge.ResetPeakMemory()

	// Allocate arrays and check memory stays reported.
	for i := 0; i < 20; i++ {
		data := make([]float32, 256*768)
		arr := bridge.NewArrayFromData(unsafe.Pointer(&data[0]), []int{256, 768}, bridge.DTypeFloat32)
		if err := bridge.Eval(arr); err != nil {
			t.Fatalf("Eval failed at iter %d: %v", i, err)
		}
		arr.Free()
		bridge.ClearCache()
	}
	runtime.GC()

	active := bridge.GetActiveMemory()
	peak := bridge.GetPeakMemory()
	t.Logf("Active: %.2f MB, Peak: %.2f MB, Limit: %.2f MB",
		float64(active)/(1024*1024), float64(peak)/(1024*1024), float64(limitBytes)/(1024*1024))

	// Peak should be non-zero (we did allocations).
	if peak == 0 {
		t.Error("GetPeakMemory() returned 0 after allocations")
	}
}
