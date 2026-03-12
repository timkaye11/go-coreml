// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// mlxBuffer wraps an MLX array as a GoMLX buffer.
// Unlike the MPSGraph backend (which copies to/from Go heap), mlxBuffer
// uses MLX's unified memory — the array IS the buffer.
type mlxBuffer struct {
	array *bridge.Array
	shape shapes.Shape
	valid bool
}

// newBufferFromArray creates a buffer wrapping an existing MLX array.
func newBufferFromArray(array *bridge.Array, shape shapes.Shape) *mlxBuffer {
	buf := &mlxBuffer{
		array: array,
		shape: shape,
		valid: true,
	}
	runtime.SetFinalizer(buf, func(b *mlxBuffer) {
		if b.array != nil {
			b.array.Free()
			b.array = nil
		}
	})
	return buf
}

// newBufferForShape creates a zero-filled buffer for the given shape.
func newBufferForShape(shape shapes.Shape, s *bridge.Stream) *mlxBuffer {
	dims := shape.Dimensions
	mlxDType := gomlxDTypeToMLX(shape.DType)
	arr := bridge.Zeros(dims, mlxDType, s)
	return newBufferFromArray(arr, shape)
}

// bufferFromFlat creates an mlxBuffer from a Go flat slice, copying the data into unified memory.
func bufferFromFlat(flat any, shape shapes.Shape) (*mlxBuffer, error) {
	flatVal := reflect.ValueOf(flat)
	if flatVal.Kind() != reflect.Slice {
		return nil, errors.Errorf("bufferFromFlat: expected slice, got %T", flat)
	}

	size := shape.Size()
	if size == 0 {
		size = 1 // Scalar
	}
	if flatVal.Len() != size {
		return nil, errors.Errorf("bufferFromFlat: expected %d elements, got %d", size, flatVal.Len())
	}

	var dataPtr unsafe.Pointer
	if flatVal.Len() > 0 {
		dataPtr = unsafe.Pointer(flatVal.Pointer())
	}

	dims := shape.Dimensions
	mlxDType := gomlxDTypeToMLX(shape.DType)
	arr := bridge.NewArrayFromData(dataPtr, dims, mlxDType)
	runtime.KeepAlive(flat)
	return newBufferFromArray(arr, shape), nil
}

// bufferCopyToFlat copies buffer data to the given flat slice.
func bufferCopyToFlat(buf *mlxBuffer, flat any) error {
	if !buf.valid {
		return errors.New("bufferCopyToFlat: buffer has been finalized")
	}

	// Ensure the array is evaluated (MLX is lazy).
	if err := bridge.Eval(buf.array); err != nil {
		return errors.Wrap(err, "bufferCopyToFlat: eval")
	}

	flatVal := reflect.ValueOf(flat)
	if flatVal.Kind() != reflect.Slice {
		return errors.Errorf("bufferCopyToFlat: expected slice, got %T", flat)
	}

	size := buf.shape.Size()
	if size == 0 {
		size = 1
	}
	if flatVal.Len() != size {
		return errors.Errorf("bufferCopyToFlat: expected %d elements, got %d", size, flatVal.Len())
	}

	// Flatten the array to ensure contiguous memory layout.
	// View operations (transpose, broadcast, etc.) may leave the data
	// non-contiguous; reshape to 1D forces a contiguous copy.
	s := bridge.DefaultGPUStream()
	defer s.Free()
	contiguous := bridge.Reshape(buf.array, []int{size}, s)
	if err := bridge.Eval(contiguous); err != nil {
		return errors.Wrap(err, "bufferCopyToFlat: flatten eval")
	}
	defer contiguous.Free()

	// Get pointer to unified memory data.
	srcPtr := contiguous.DataPtr()
	if srcPtr == nil {
		return errors.New("bufferCopyToFlat: nil data pointer from MLX array")
	}

	// Copy from unified memory to Go slice.
	nbytes := contiguous.NBytes()
	dstPtr := unsafe.Pointer(flatVal.Pointer())
	dstSlice := unsafe.Slice((*byte)(dstPtr), nbytes)
	srcSlice := unsafe.Slice((*byte)(srcPtr), nbytes)
	copy(dstSlice, srcSlice)

	return nil
}

// gomlxDTypeToMLX converts a GoMLX dtype to MLX bridge dtype.
func gomlxDTypeToMLX(dt dtypes.DType) bridge.DType {
	switch dt {
	case dtypes.Bool:
		return bridge.DTypeBool
	case dtypes.Int8:
		return bridge.DTypeInt8
	case dtypes.Int16:
		return bridge.DTypeInt16
	case dtypes.Int32:
		return bridge.DTypeInt32
	case dtypes.Int64:
		return bridge.DTypeInt64
	case dtypes.Uint8:
		return bridge.DTypeUint8
	case dtypes.Uint16:
		return bridge.DTypeUint16
	case dtypes.Uint32:
		return bridge.DTypeUint32
	case dtypes.Uint64:
		return bridge.DTypeUint64
	case dtypes.Float16:
		return bridge.DTypeFloat16
	case dtypes.BFloat16:
		return bridge.DTypeBFloat16
	case dtypes.Float32:
		return bridge.DTypeFloat32
	case dtypes.Float64:
		return bridge.DTypeFloat64
	case dtypes.Complex64:
		return bridge.DTypeComplex64
	default:
		panic(fmt.Sprintf("gomlxDTypeToMLX: unsupported dtype %v", dt))
	}
}

// mlxDTypeToGoMLX converts an MLX bridge dtype to a GoMLX dtype.
func mlxDTypeToGoMLX(dt bridge.DType) dtypes.DType {
	switch dt {
	case bridge.DTypeBool:
		return dtypes.Bool
	case bridge.DTypeInt8:
		return dtypes.Int8
	case bridge.DTypeInt16:
		return dtypes.Int16
	case bridge.DTypeInt32:
		return dtypes.Int32
	case bridge.DTypeInt64:
		return dtypes.Int64
	case bridge.DTypeUint8:
		return dtypes.Uint8
	case bridge.DTypeUint16:
		return dtypes.Uint16
	case bridge.DTypeUint32:
		return dtypes.Uint32
	case bridge.DTypeUint64:
		return dtypes.Uint64
	case bridge.DTypeFloat16:
		return dtypes.Float16
	case bridge.DTypeBFloat16:
		return dtypes.BFloat16
	case bridge.DTypeFloat32:
		return dtypes.Float32
	case bridge.DTypeFloat64:
		return dtypes.Float64
	case bridge.DTypeComplex64:
		return dtypes.Complex64
	default:
		panic(fmt.Sprintf("mlxDTypeToGoMLX: unsupported dtype %v", dt))
	}
}
