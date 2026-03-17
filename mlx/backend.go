// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package mlx implements a GoMLX backend using Apple's MLX framework
// for GPU-accelerated tensor computation on Apple Silicon, with support
// for automatic differentiation, JIT compilation, and unified memory.
package mlx

/*
#include <sys/sysctl.h>
#include <stdint.h>

static uint64_t get_physical_memory() {
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    sysctl(mib, 2, &mem, &len, NULL, 0);
    return mem;
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Ensure unsafe is used.
var _ = unsafe.Pointer(nil)

// BackendName is the name used to register this backend.
const BackendName = "mlx"

// Backend implements the backends.Backend interface using Apple's MLX.
type Backend struct {
	gpuStream   *bridge.Stream
	cpuStream   *bridge.Stream
	mu          sync.RWMutex
	isFinalized bool
}

// Verify interface compliance.
var _ backends.Backend = &Backend{}

// DefaultMemoryLimitFraction is the fraction of system memory MLX is allowed to use.
// This must be below ~0.70 (0.95 * recommendedMaxWorkingSetSize / totalMem) to actually
// affect MLX's internal gc_limit. Values above the hardware-clamped threshold are no-ops.
// On a 24GB M4 Pro: 0.55 * 24 = 13.2 GB, safely below the 16.87 GB hardware clamp.
const DefaultMemoryLimitFraction = 0.55

// DefaultCacheLimitFraction is the fraction of the memory limit used for MLX's cache.
// Lower values force more aggressive buffer release after each operation.
// On a 24GB M4 Pro: 0.25 * 13.2 GB = 3.3 GB cache limit.
const DefaultCacheLimitFraction = 0.25

// New creates a new MLX backend.
func New(config string) (backends.Backend, error) {
	if !bridge.MetalIsAvailable() {
		return nil, errors.New("MLX backend requires Metal GPU (Apple Silicon)")
	}
	b := &Backend{
		gpuStream: bridge.DefaultGPUStream(),
		cpuStream: bridge.DefaultCPUStream(),
	}

	// Set memory limits to prevent unbounded memory growth that can crash the kernel.
	// Without limits, MLX will consume all available unified memory, causing the
	// macOS watchdog to trigger a kernel panic when swap is exhausted.
	totalMem := systemMemoryBytes()
	if totalMem > 0 {
		memLimit := uint64(float64(totalMem) * DefaultMemoryLimitFraction)
		cacheLimit := uint64(float64(memLimit) * DefaultCacheLimitFraction)
		bridge.SetMemoryLimit(memLimit)
		bridge.SetCacheLimit(cacheLimit)
		fmt.Printf("  MLX memory limit: %.1f GB (cache: %.1f GB) of %.1f GB total\n",
			float64(memLimit)/(1<<30), float64(cacheLimit)/(1<<30), float64(totalMem)/(1<<30))
	}

	return b, nil
}

// ClearCache frees cached GPU memory. Call periodically during long training runs
// to prevent memory accumulation.
func (b *Backend) ClearCache() {
	bridge.ClearCache()
}

// GetActiveMemory returns current GPU memory allocation in bytes.
func (b *Backend) GetActiveMemory() uint64 {
	return bridge.GetActiveMemory()
}

// GetCacheMemory returns current GPU cache memory in bytes.
func (b *Backend) GetCacheMemory() uint64 {
	return bridge.GetCacheMemory()
}

// GetPeakMemory returns peak GPU memory usage since last reset.
func (b *Backend) GetPeakMemory() uint64 {
	return bridge.GetPeakMemory()
}

// ResetPeakMemory resets the peak memory counter.
func (b *Backend) ResetPeakMemory() {
	bridge.ResetPeakMemory()
}

// SystemMemoryBytes returns total physical memory in bytes.
func (b *Backend) SystemMemoryBytes() uint64 {
	return systemMemoryBytes()
}

// Name returns the backend name.
func (b *Backend) Name() string { return BackendName }

// String returns the backend name.
func (b *Backend) String() string { return b.Name() }

// Description returns a human-readable description.
func (b *Backend) Description() string {
	return "MLX GPU backend (Apple Silicon, unified memory)"
}

// NumDevices returns 1 (single GPU on Apple Silicon).
func (b *Backend) NumDevices() int { return 1 }

// DeviceDescription returns the device description.
func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	return "Apple Silicon GPU (MLX)"
}

// Capabilities returns the set of supported ops and dtypes.
func (b *Backend) Capabilities() backends.Capabilities {
	return backendCapabilities
}

// Builder creates a new computation builder.
func (b *Backend) Builder(name string) backends.Builder {
	return newBuilder(b, name)
}

// Finalize releases all backend resources.
func (b *Backend) Finalize() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFinalized {
		return
	}
	b.isFinalized = true
	if b.gpuStream != nil {
		b.gpuStream.Free()
		b.gpuStream = nil
	}
	if b.cpuStream != nil {
		b.cpuStream.Free()
		b.cpuStream = nil
	}
	bridge.ClearCache()
}

// IsFinalized returns whether the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.isFinalized
}

// stream returns the GPU stream for operations.
func (b *Backend) stream() *bridge.Stream {
	return b.gpuStream
}

// --- DataInterface implementation ---

// BufferFinalize releases a buffer's resources.
func (b *Backend) BufferFinalize(buffer backends.Buffer) error {
	if b.isFinalized {
		return nil
	}
	buf, ok := buffer.(*mlxBuffer)
	if !ok {
		return errors.Errorf("BufferFinalize: expected *mlxBuffer, got %T", buffer)
	}
	buf.valid = false
	runtime.SetFinalizer(buf, nil) // Cancel GC finalizer to prevent double-free.
	if buf.array != nil {
		buf.array.Free()
		buf.array = nil
	}
	return nil
}

// BufferShape returns the shape of a buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	buf, ok := buffer.(*mlxBuffer)
	if !ok {
		return shapes.Invalid(), errors.Errorf("BufferShape: expected *mlxBuffer, got %T", buffer)
	}
	return buf.shape, nil
}

// BufferDeviceNum returns the device number (always 0 for single device).
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	return 0, nil
}

// BufferToFlatData copies buffer data into a Go flat slice.
func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) error {
	buf, ok := buffer.(*mlxBuffer)
	if !ok {
		return errors.Errorf("BufferToFlatData: expected *mlxBuffer, got %T", buffer)
	}
	return bufferCopyToFlat(buf, flat)
}

// BufferFromFlatData creates a buffer from a Go flat slice.
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	return bufferFromFlat(flat, shape)
}

// HasSharedBuffers returns true since MLX uses unified memory.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer creates a new buffer with direct access to its memory.
func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any, err error) {
	buf := newBufferForShape(shape, b.gpuStream)
	// Evaluate to materialize in unified memory.
	if evalErr := bridge.Eval(buf.array); evalErr != nil {
		return nil, nil, errors.Wrap(evalErr, "NewSharedBuffer: eval")
	}
	// Return the flat data pointing to unified memory.
	flatData := bufferToFlatSlice(buf)
	return buf, flatData, nil
}

// BufferData returns direct access to the buffer's flat data.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	buf, ok := buffer.(*mlxBuffer)
	if !ok {
		return nil, errors.Errorf("BufferData: expected *mlxBuffer, got %T", buffer)
	}
	// Ensure evaluated.
	if evalErr := bridge.Eval(buf.array); evalErr != nil {
		return nil, errors.Wrap(evalErr, "BufferData: eval")
	}
	return bufferToFlatSlice(buf), nil
}

// BufferCopyToDevice copies buffer to another device (not supported with single device).
func (b *Backend) BufferCopyToDevice(source backends.Buffer, deviceNum backends.DeviceNum) (backends.Buffer, error) {
	return nil, errors.New("BufferCopyToDevice: only one device supported")
}

// systemMemoryBytes returns total physical memory in bytes via sysctl.
func systemMemoryBytes() uint64 {
	return uint64(C.get_physical_memory())
}

// bufferToFlatSlice creates a Go slice backed by the MLX array's unified memory.
func bufferToFlatSlice(buf *mlxBuffer) any {
	ptr := buf.array.DataPtr()
	size := buf.shape.Size()
	if size == 0 {
		size = 1
	}
	return makeSliceFromPtr(ptr, size, buf.shape.DType)
}

// makeSliceFromPtr creates a Go slice pointing to existing memory.
func makeSliceFromPtr(ptr unsafe.Pointer, length int, dt dtypes.DType) any {
	switch dt {
	case dtypes.Float32:
		return unsafe.Slice((*float32)(ptr), length)
	case dtypes.Float64:
		return unsafe.Slice((*float64)(ptr), length)
	case dtypes.Int8:
		return unsafe.Slice((*int8)(ptr), length)
	case dtypes.Int16:
		return unsafe.Slice((*int16)(ptr), length)
	case dtypes.Int32:
		return unsafe.Slice((*int32)(ptr), length)
	case dtypes.Int64:
		return unsafe.Slice((*int64)(ptr), length)
	case dtypes.Uint8:
		return unsafe.Slice((*uint8)(ptr), length)
	case dtypes.Uint16:
		return unsafe.Slice((*uint16)(ptr), length)
	case dtypes.Uint32:
		return unsafe.Slice((*uint32)(ptr), length)
	case dtypes.Uint64:
		return unsafe.Slice((*uint64)(ptr), length)
	case dtypes.Bool:
		return unsafe.Slice((*bool)(ptr), length)
	default:
		// For Float16/BFloat16/Complex64, return as byte slice
		elemSize := dt.Size()
		return unsafe.Slice((*byte)(ptr), length*elemSize)
	}
}
