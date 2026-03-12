// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package bridge provides CGo bindings to Apple's MLX framework via mlx-c
// for GPU-accelerated tensor computation on Apple Silicon.
package bridge

/*
#cgo darwin CFLAGS: -I${SRCDIR}/deps/include
#cgo darwin LDFLAGS: -L${SRCDIR}/deps/lib -lmlxc -lmlx -lc++ -framework Metal -framework Foundation -framework Accelerate

#include "mlx/c/mlx.h"
#include "mlx/c/compile.h"
#include <stdlib.h>
#include <string.h>

// Wrapper to avoid CGo issues with __fp16 type.
static const void* mlx_array_data_float16_ptr(mlx_array arr) {
    return (const void*)mlx_array_data_float16(arr);
}

// Helper: check if mlx_array has a valid context.
static int mlx_array_ctx_is_null(mlx_array arr) {
    return arr.ctx == NULL;
}
static int mlx_stream_ctx_is_null(mlx_stream s) {
    return s.ctx == NULL;
}
static int mlx_vector_array_ctx_is_null(mlx_vector_array v) {
    return v.ctx == NULL;
}

// CGo trampoline: C function that forwards to Go callback via payload.
extern int goClosureCallback(mlx_vector_array* res, const mlx_vector_array inputs, void* payload);

static mlx_closure create_go_closure(void* payload) {
    return mlx_closure_new_func_payload(goClosureCallback, payload, NULL);
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// DType mirrors MLX dtype enum.
type DType = C.mlx_dtype

var (
	DTypeBool      DType = C.MLX_BOOL
	DTypeUint8     DType = C.MLX_UINT8
	DTypeUint16    DType = C.MLX_UINT16
	DTypeUint32    DType = C.MLX_UINT32
	DTypeUint64    DType = C.MLX_UINT64
	DTypeInt8      DType = C.MLX_INT8
	DTypeInt16     DType = C.MLX_INT16
	DTypeInt32     DType = C.MLX_INT32
	DTypeInt64     DType = C.MLX_INT64
	DTypeFloat16   DType = C.MLX_FLOAT16
	DTypeBFloat16  DType = C.MLX_BFLOAT16
	DTypeFloat32   DType = C.MLX_FLOAT32
	DTypeFloat64   DType = C.MLX_FLOAT64
	DTypeComplex64 DType = C.MLX_COMPLEX64
)

// ReduceType for reduction operations.
const (
	ReduceSum     = 0
	ReduceProduct = 1
	ReduceMax     = 2
	ReduceMin     = 3
)

// ScatterMode for scatter operations.
const (
	ScatterModeAdd = 0
	ScatterModeMax = 1
	ScatterModeMin = 2
)

// checkRC checks an mlx-c return code and panics on error.
func checkRC(rc C.int, op string) {
	if rc != 0 {
		panic(fmt.Sprintf("mlx: %s failed with rc=%d", op, rc))
	}
}

// checkRCErr checks an mlx-c return code and returns an error.
func checkRCErr(rc C.int, op string) error {
	if rc != 0 {
		return fmt.Errorf("mlx: %s failed with rc=%d", op, rc)
	}
	return nil
}

// ===========================================================================
// Array
// ===========================================================================

// Array wraps an mlx_array handle.
type Array struct {
	handle C.mlx_array
}

func (a *Array) valid() bool {
	return C.mlx_array_ctx_is_null(a.handle) == 0
}

// NewArray creates an empty array.
func NewArray() *Array {
	a := &Array{}
	a.handle = C.mlx_array_new()
	return a
}

// NewArrayFromData creates an array from raw data.
func NewArrayFromData(data unsafe.Pointer, shape []int, dtype DType) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	if data == nil {
		// Create zeros array instead.
		rc := C.mlx_zeros(&a.handle, shapePtr, C.size_t(len(shape)), dtype, C.mlx_default_gpu_stream_new())
		checkRC(rc, "mlx_zeros")
	} else {
		a.handle = C.mlx_array_new_data(data, shapePtr, C.int(len(shape)), dtype)
	}
	runtime.KeepAlive(data)
	return a
}

// NewArrayScalarFloat32 creates a scalar float32 array.
func NewArrayScalarFloat32(val float32) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_float(C.float(val))
	return a
}

// NewArrayScalarInt32 creates a scalar int32 array.
func NewArrayScalarInt32(val int32) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_int(C.int(val))
	return a
}

// NewArrayScalarBool creates a scalar bool array.
func NewArrayScalarBool(val bool) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_bool(C.bool(val))
	return a
}

// Free releases the array.
func (a *Array) Free() {
	if a.valid() {
		C.mlx_array_free(a.handle)
		a.handle = C.mlx_array_new() // reset to empty
	}
}

// Shape returns the dimensions.
func (a *Array) Shape() []int {
	ndim := int(C.mlx_array_ndim(a.handle))
	shape := make([]int, ndim)
	cShape := C.mlx_array_shape(a.handle)
	for i := 0; i < ndim; i++ {
		shape[i] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(cShape)) + uintptr(i)*unsafe.Sizeof(C.int(0)))))
	}
	return shape
}

// NDim returns the number of dimensions.
func (a *Array) NDim() int {
	return int(C.mlx_array_ndim(a.handle))
}

// Size returns the total number of elements.
func (a *Array) Size() int {
	return int(C.mlx_array_size(a.handle))
}

// NBytes returns the total byte count.
func (a *Array) NBytes() int {
	return int(C.mlx_array_nbytes(a.handle))
}

// DType returns the element type.
func (a *Array) DType() DType {
	return C.mlx_array_dtype(a.handle)
}

// ItemSize returns the size of each element in bytes.
func (a *Array) ItemSize() int {
	return int(C.mlx_array_itemsize(a.handle))
}

// DataPtr returns a pointer to the array's data in unified memory.
// The array must be evaluated first (call Eval).
func (a *Array) DataPtr() unsafe.Pointer {
	dtype := a.DType()
	switch dtype {
	case DTypeBool:
		return unsafe.Pointer(C.mlx_array_data_bool(a.handle))
	case DTypeUint8:
		return unsafe.Pointer(C.mlx_array_data_uint8(a.handle))
	case DTypeUint16:
		return unsafe.Pointer(C.mlx_array_data_uint16(a.handle))
	case DTypeUint32:
		return unsafe.Pointer(C.mlx_array_data_uint32(a.handle))
	case DTypeUint64:
		return unsafe.Pointer(C.mlx_array_data_uint64(a.handle))
	case DTypeInt8:
		return unsafe.Pointer(C.mlx_array_data_int8(a.handle))
	case DTypeInt16:
		return unsafe.Pointer(C.mlx_array_data_int16(a.handle))
	case DTypeInt32:
		return unsafe.Pointer(C.mlx_array_data_int32(a.handle))
	case DTypeInt64:
		return unsafe.Pointer(C.mlx_array_data_int64(a.handle))
	case DTypeFloat16:
		return unsafe.Pointer(C.mlx_array_data_float16_ptr(a.handle))
	case DTypeFloat32:
		return unsafe.Pointer(C.mlx_array_data_float32(a.handle))
	case DTypeFloat64:
		return unsafe.Pointer(C.mlx_array_data_float64(a.handle))
	default:
		return unsafe.Pointer(C.mlx_array_data_float32(a.handle))
	}
}

// ===========================================================================
// Vector of Arrays (for multi-input/output ops)
// ===========================================================================

// VectorArray wraps mlx_vector_array.
type VectorArray struct {
	handle C.mlx_vector_array
}

func (v *VectorArray) valid() bool {
	return C.mlx_vector_array_ctx_is_null(v.handle) == 0
}

// NewVectorArray creates a new vector from a slice of arrays.
func NewVectorArray(arrays []*Array) *VectorArray {
	v := &VectorArray{}
	if len(arrays) == 0 {
		v.handle = C.mlx_vector_array_new()
		return v
	}
	handles := make([]C.mlx_array, len(arrays))
	for i, a := range arrays {
		handles[i] = a.handle
	}
	v.handle = C.mlx_vector_array_new_data(&handles[0], C.size_t(len(arrays)))
	return v
}

// Free releases the vector.
func (v *VectorArray) Free() {
	if v.valid() {
		C.mlx_vector_array_free(v.handle)
		v.handle = C.mlx_vector_array_new()
	}
}

// Size returns the number of arrays in the vector.
func (v *VectorArray) Size() int {
	return int(C.mlx_vector_array_size(v.handle))
}

// Get returns the array at the given index. Caller must free the returned array.
func (v *VectorArray) Get(index int) *Array {
	a := &Array{}
	rc := C.mlx_vector_array_get(&a.handle, v.handle, C.size_t(index))
	checkRC(rc, "mlx_vector_array_get")
	return a
}

// ===========================================================================
// Stream
// ===========================================================================

// Stream wraps mlx_stream.
type Stream struct {
	handle C.mlx_stream
}

func (s *Stream) valid() bool {
	return C.mlx_stream_ctx_is_null(s.handle) == 0
}

// DefaultGPUStream returns the default GPU stream.
func DefaultGPUStream() *Stream {
	s := &Stream{}
	s.handle = C.mlx_default_gpu_stream_new()
	return s
}

// DefaultCPUStream returns the default CPU stream.
func DefaultCPUStream() *Stream {
	s := &Stream{}
	s.handle = C.mlx_default_cpu_stream_new()
	return s
}

// Free releases the stream.
func (s *Stream) Free() {
	if s.valid() {
		C.mlx_stream_free(s.handle)
		s.handle = C.mlx_stream_new()
	}
}

// ===========================================================================
// Eval
// ===========================================================================

// Eval forces evaluation of arrays. MLX is lazy — this materializes results.
func Eval(arrays ...*Array) error {
	if len(arrays) == 0 {
		return nil
	}
	v := NewVectorArray(arrays)
	defer v.Free()
	rc := C.mlx_eval(v.handle)
	return checkRCErr(rc, "mlx_eval")
}

// ===========================================================================
// Device Queries
// ===========================================================================

// MetalIsAvailable returns true if Metal GPU is available.
func MetalIsAvailable() bool {
	var result C.bool
	C.mlx_metal_is_available(&result)
	return bool(result)
}

// ===========================================================================
// Memory Management
// ===========================================================================

// ClearCache frees cached GPU memory.
func ClearCache() {
	C.mlx_clear_cache()
}

// GetActiveMemory returns current GPU memory allocation in bytes.
func GetActiveMemory() uint64 {
	var res C.size_t
	C.mlx_get_active_memory(&res)
	return uint64(res)
}

// SetMemoryLimit sets the maximum GPU memory limit.
func SetMemoryLimit(limit uint64) uint64 {
	var res C.size_t
	C.mlx_set_memory_limit(&res, C.size_t(limit))
	return uint64(res)
}

// SetCacheLimit sets the maximum GPU cache limit.
func SetCacheLimit(limit uint64) uint64 {
	var res C.size_t
	C.mlx_set_cache_limit(&res, C.size_t(limit))
	return uint64(res)
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func Abs(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_abs(&r.handle, x.handle, s.handle), "mlx_abs"); return r
}
func Negative(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_negative(&r.handle, x.handle, s.handle), "mlx_negative"); return r
}
func Sqrt(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sqrt(&r.handle, x.handle, s.handle), "mlx_sqrt"); return r
}
func Rsqrt(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_rsqrt(&r.handle, x.handle, s.handle), "mlx_rsqrt"); return r
}
func Exp(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_exp(&r.handle, x.handle, s.handle), "mlx_exp"); return r
}
func Expm1(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_expm1(&r.handle, x.handle, s.handle), "mlx_expm1"); return r
}
func Log(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_log(&r.handle, x.handle, s.handle), "mlx_log"); return r
}
func Log1p(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_log1p(&r.handle, x.handle, s.handle), "mlx_log1p"); return r
}
func Sin(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sin(&r.handle, x.handle, s.handle), "mlx_sin"); return r
}
func Cos(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_cos(&r.handle, x.handle, s.handle), "mlx_cos"); return r
}
func Tanh(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_tanh(&r.handle, x.handle, s.handle), "mlx_tanh"); return r
}
func Sigmoid(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sigmoid(&r.handle, x.handle, s.handle), "mlx_sigmoid"); return r
}
func Erf(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_erf(&r.handle, x.handle, s.handle), "mlx_erf"); return r
}
func Floor(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_floor(&r.handle, x.handle, s.handle), "mlx_floor"); return r
}
func Ceil(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_ceil(&r.handle, x.handle, s.handle), "mlx_ceil"); return r
}
func Sign(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sign(&r.handle, x.handle, s.handle), "mlx_sign"); return r
}
func LogicalNot(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_not(&r.handle, x.handle, s.handle), "mlx_logical_not"); return r
}
func IsNaN(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_isnan(&r.handle, x.handle, s.handle), "mlx_isnan"); return r
}
func BitwiseNot(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_invert(&r.handle, x.handle, s.handle), "mlx_bitwise_invert"); return r
}
func Round(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_round(&r.handle, x.handle, 0, s.handle), "mlx_round"); return r
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func Add(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_add(&r.handle, a.handle, b.handle, s.handle), "mlx_add"); return r
}
func Subtract(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_subtract(&r.handle, a.handle, b.handle, s.handle), "mlx_subtract"); return r
}
func Multiply(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_multiply(&r.handle, a.handle, b.handle, s.handle), "mlx_multiply"); return r
}
func Divide(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_divide(&r.handle, a.handle, b.handle, s.handle), "mlx_divide"); return r
}
func Remainder(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_remainder(&r.handle, a.handle, b.handle, s.handle), "mlx_remainder"); return r
}
func Power(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_power(&r.handle, a.handle, b.handle, s.handle), "mlx_power"); return r
}
func Maximum(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_maximum(&r.handle, a.handle, b.handle, s.handle), "mlx_maximum"); return r
}
func Minimum(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_minimum(&r.handle, a.handle, b.handle, s.handle), "mlx_minimum"); return r
}
func Arctan2(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_arctan2(&r.handle, a.handle, b.handle, s.handle), "mlx_arctan2"); return r
}
func LogicalAnd(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_and(&r.handle, a.handle, b.handle, s.handle), "mlx_logical_and"); return r
}
func LogicalOr(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_or(&r.handle, a.handle, b.handle, s.handle), "mlx_logical_or"); return r
}
func BitwiseAnd(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_and(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_and"); return r
}
func BitwiseOr(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_or(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_or"); return r
}
func BitwiseXor(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_xor(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_xor"); return r
}
func LeftShift(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_left_shift(&r.handle, a.handle, b.handle, s.handle), "mlx_left_shift"); return r
}
func RightShift(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_right_shift(&r.handle, a.handle, b.handle, s.handle), "mlx_right_shift"); return r
}
func Equal(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_equal"); return r
}
func NotEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_not_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_not_equal"); return r
}
func Less(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_less(&r.handle, a.handle, b.handle, s.handle), "mlx_less"); return r
}
func LessEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_less_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_less_equal"); return r
}
func Greater(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_greater(&r.handle, a.handle, b.handle, s.handle), "mlx_greater"); return r
}
func GreaterEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_greater_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_greater_equal"); return r
}

// ===========================================================================
// Shape Operations
// ===========================================================================

// Reshape reshapes the array.
func Reshape(x *Array, shape []int, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_reshape(&r.handle, x.handle, shapePtr, C.size_t(len(shape)), s.handle)
	checkRC(rc, "mlx_reshape")
	return r
}

// Transpose permutes axes. Empty axes reverses all axes.
func Transpose(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	if len(axes) == 0 {
		rc := C.mlx_transpose(&r.handle, x.handle, s.handle)
		checkRC(rc, "mlx_transpose")
		return r
	}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_transpose_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_transpose_axes")
	return r
}

// BroadcastTo broadcasts to the target shape.
func BroadcastTo(x *Array, shape []int, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	rc := C.mlx_broadcast_to(&r.handle, x.handle, &cShape[0], C.size_t(len(shape)), s.handle)
	checkRC(rc, "mlx_broadcast_to")
	return r
}

// AsType casts the array to a different dtype.
func AsType(x *Array, dtype DType, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_astype(&r.handle, x.handle, dtype, s.handle)
	checkRC(rc, "mlx_astype")
	return r
}

// Concatenate joins arrays along an axis.
func Concatenate(arrays []*Array, axis int, s *Stream) *Array {
	r := &Array{}
	v := NewVectorArray(arrays)
	defer v.Free()
	rc := C.mlx_concatenate_axis(&r.handle, v.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_concatenate_axis")
	return r
}

// Slice extracts a sub-array.
func Slice(x *Array, starts, stops, strides []int, s *Stream) *Array {
	r := &Array{}
	n := len(starts)
	cStarts := make([]C.int, n)
	cStops := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStarts[i] = C.int(starts[i])
		cStops[i] = C.int(stops[i])
		cStrides[i] = C.int(strides[i])
	}
	rc := C.mlx_slice(&r.handle, x.handle, &cStarts[0], C.size_t(n), &cStops[0], C.size_t(n), &cStrides[0], C.size_t(n), s.handle)
	checkRC(rc, "mlx_slice")
	return r
}

// SliceUpdate updates a slice of an array.
func SliceUpdate(src, update *Array, starts, stops, strides []int, s *Stream) *Array {
	r := &Array{}
	n := len(starts)
	cStarts := make([]C.int, n)
	cStops := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStarts[i] = C.int(starts[i])
		cStops[i] = C.int(stops[i])
		cStrides[i] = C.int(strides[i])
	}
	rc := C.mlx_slice_update(&r.handle, src.handle, update.handle, &cStarts[0], C.size_t(n), &cStops[0], C.size_t(n), &cStrides[0], C.size_t(n), s.handle)
	checkRC(rc, "mlx_slice_update")
	return r
}

// Pad pads the array.
func Pad(x, padValue *Array, axes []int, lowPads, highPads []int, s *Stream) *Array {
	r := &Array{}
	n := len(axes)
	cAxes := make([]C.int, n)
	cLow := make([]C.int, n)
	cHigh := make([]C.int, n)
	for i := 0; i < n; i++ {
		cAxes[i] = C.int(axes[i])
		cLow[i] = C.int(lowPads[i])
		cHigh[i] = C.int(highPads[i])
	}
	cMode := C.CString("constant")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_pad(&r.handle, x.handle, &cAxes[0], C.size_t(n), &cLow[0], C.size_t(n), &cHigh[0], C.size_t(n), padValue.handle, cMode, s.handle)
	checkRC(rc, "mlx_pad")
	return r
}

// Flip reverses elements along the given axes.
func Flip(x *Array, axes []int, s *Stream) *Array {
	result := x
	shape := x.Shape()
	for _, axis := range axes {
		n := shape[axis]
		// Create reversed index array: [n-1, n-2, ..., 1, 0]
		indices := Arange(float64(n-1), -1, -1, C.MLX_INT32, s)
		result = Take(result, indices, axis, s)
	}
	return result
}

// ExpandDims adds dimensions at the given axes.
func ExpandDims(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_expand_dims_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_expand_dims_axes")
	return r
}

// Squeeze removes dimensions at the given axes.
func Squeeze(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_squeeze_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_squeeze_axes")
	return r
}

// Where selects elements based on condition.
func Where(condition, x, y *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_where(&r.handle, condition.handle, x.handle, y.handle, s.handle)
	checkRC(rc, "mlx_where")
	return r
}

// Clip clamps values to a range.
func Clip(x, min, max *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_clip(&r.handle, x.handle, min.handle, max.handle, s.handle)
	checkRC(rc, "mlx_clip")
	return r
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

func reduceSetup(axes []int) ([]C.int, *C.int, C.size_t) {
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	if len(cAxes) > 0 {
		return cAxes, &cAxes[0], C.size_t(len(axes))
	}
	return cAxes, nil, 0
}

func Sum(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_sum_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_sum_axes"); runtime.KeepAlive(ca); return r
}
func Prod(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_prod_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_prod_axes"); runtime.KeepAlive(ca); return r
}
func Max(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_max_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_max_axes"); runtime.KeepAlive(ca); return r
}
func Min(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_min_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_min_axes"); runtime.KeepAlive(ca); return r
}
func All(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_all_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_all_axes"); runtime.KeepAlive(ca); return r
}
func Any(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_any_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_any_axes"); runtime.KeepAlive(ca); return r
}

// ArgMin returns indices of minimum values along an axis.
func ArgMin(x *Array, axis int, keepDims bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argmin_axis(&r.handle, x.handle, C.int(axis), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_argmin_axis")
	return r
}

// ArgMax returns indices of maximum values along an axis.
func ArgMax(x *Array, axis int, keepDims bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argmax_axis(&r.handle, x.handle, C.int(axis), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_argmax_axis")
	return r
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

// MatMul performs matrix multiplication.
func MatMul(lhs, rhs *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_matmul(&r.handle, lhs.handle, rhs.handle, s.handle)
	checkRC(rc, "mlx_matmul")
	return r
}

// ===========================================================================
// Gather / Scatter
// ===========================================================================

// Take gathers elements along an axis.
func Take(x, indices *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_take_axis(&r.handle, x.handle, indices.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_take_axis")
	return r
}

// TakeAlongAxis gathers elements along an axis using indices.
func TakeAlongAxis(x, indices *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_take_along_axis(&r.handle, x.handle, indices.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_take_along_axis")
	return r
}

// Gather performs generalized gather.
func Gather(x *Array, indices []*Array, axes []int, sliceSizes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	cSliceSizes := make([]C.int, len(sliceSizes))
	for i, sz := range sliceSizes {
		cSliceSizes[i] = C.int(sz)
	}
	rc := C.mlx_gather(&r.handle, x.handle, indicesVec.handle, &cAxes[0], C.size_t(len(axes)), &cSliceSizes[0], C.size_t(len(sliceSizes)), s.handle)
	checkRC(rc, "mlx_gather")
	return r
}

// ScatterAdd performs scatter with addition.
func ScatterAdd(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_add(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_add")
	return r
}

// ScatterMax performs scatter with max.
func ScatterMax(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_max(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_max")
	return r
}

// ScatterMin performs scatter with min.
func ScatterMin(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_min(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_min")
	return r
}

// ===========================================================================
// Convolution
// ===========================================================================

// Conv2d performs 2D convolution.
func Conv2d(input, weight *Array, stride, padding, dilation [2]int, groups int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_conv2d(&r.handle, input.handle, weight.handle,
		C.int(stride[0]), C.int(stride[1]),
		C.int(padding[0]), C.int(padding[1]),
		C.int(dilation[0]), C.int(dilation[1]),
		C.int(groups), s.handle)
	checkRC(rc, "mlx_conv2d")
	return r
}

// Conv1d performs 1D convolution.
func Conv1d(input, weight *Array, stride, padding, dilation, groups int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_conv1d(&r.handle, input.handle, weight.handle,
		C.int(stride), C.int(padding), C.int(dilation), C.int(groups), s.handle)
	checkRC(rc, "mlx_conv1d")
	return r
}

// ===========================================================================
// Softmax
// ===========================================================================

// Softmax computes softmax along an axis.
func Softmax(x *Array, axis int, precise bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_softmax_axis(&r.handle, x.handle, C.int(axis), C.bool(precise), s.handle)
	checkRC(rc, "mlx_softmax_axis")
	return r
}

// ===========================================================================
// Sort / ArgSort
// ===========================================================================

// Sort sorts along an axis.
func Sort(x *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_sort_axis(&r.handle, x.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_sort_axis")
	return r
}

// ArgSort returns sort indices along an axis.
func ArgSort(x *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argsort_axis(&r.handle, x.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_argsort_axis")
	return r
}

// ===========================================================================
// Arange (for Iota implementation)
// ===========================================================================

// Arange creates a range of values.
func Arange(start, stop, step float64, dtype DType, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_arange(&r.handle, C.double(start), C.double(stop), C.double(step), dtype, s.handle)
	checkRC(rc, "mlx_arange")
	return r
}

// ===========================================================================
// Random Number Generation
// ===========================================================================

// RandomKey creates an RNG key from a seed.
func RandomKey(seed uint64) *Array {
	r := &Array{}
	rc := C.mlx_random_key(&r.handle, C.uint64_t(seed))
	checkRC(rc, "mlx_random_key")
	return r
}

// RandomSplit splits an RNG key into two.
func RandomSplit(key *Array, s *Stream) (*Array, *Array) {
	k1 := &Array{}
	k2 := &Array{}
	rc := C.mlx_random_split(&k1.handle, &k2.handle, key.handle, s.handle)
	checkRC(rc, "mlx_random_split")
	return k1, k2
}

// RandomUniform generates uniform random values in [low, high).
func RandomUniform(low, high *Array, shape []int, dtype DType, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_uniform(&r.handle, low.handle, high.handle, shapePtr, C.size_t(len(shape)), dtype, key.handle, s.handle)
	checkRC(rc, "mlx_random_uniform")
	return r
}

// RandomNormal generates normally distributed random values.
func RandomNormal(shape []int, dtype DType, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_normal(&r.handle, shapePtr, C.size_t(len(shape)), dtype, C.float(0), C.float(1), key.handle, s.handle)
	checkRC(rc, "mlx_random_normal")
	return r
}

// RandomBits generates random bits.
func RandomBits(shape []int, width int, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_bits(&r.handle, shapePtr, C.size_t(len(shape)), C.int(width), key.handle, s.handle)
	checkRC(rc, "mlx_random_bits")
	return r
}

// ===========================================================================
// Fast Operations
// ===========================================================================

// FastLayerNorm computes layer normalization using optimized Metal kernels.
func FastLayerNorm(x, weight, bias *Array, eps float32, s *Stream) *Array {
	r := &Array{}
	var w, b C.mlx_array
	if weight != nil {
		w = weight.handle
	}
	if bias != nil {
		b = bias.handle
	}
	rc := C.mlx_fast_layer_norm(&r.handle, x.handle, w, b, C.float(eps), s.handle)
	checkRC(rc, "mlx_fast_layer_norm")
	return r
}

// FastRMSNorm computes RMS normalization using optimized Metal kernels.
func FastRMSNorm(x, weight *Array, eps float32, s *Stream) *Array {
	r := &Array{}
	var w C.mlx_array
	if weight != nil {
		w = weight.handle
	}
	rc := C.mlx_fast_rms_norm(&r.handle, x.handle, w, C.float(eps), s.handle)
	checkRC(rc, "mlx_fast_rms_norm")
	return r
}

// FastRope computes rotary position embeddings using optimized Metal kernels.
func FastRope(x *Array, dims int, traditional bool, base float32, scale float32, offset int, s *Stream) *Array {
	r := &Array{}
	optBase := C.mlx_optional_float{value: C.float(base), has_value: true}
	var freqs C.mlx_array // null
	rc := C.mlx_fast_rope(&r.handle, x.handle, C.int(dims), C.bool(traditional),
		optBase, C.float(scale), C.int(offset), freqs, s.handle)
	checkRC(rc, "mlx_fast_rope")
	return r
}

// FastScaledDotProductAttention computes scaled dot product attention.
func FastScaledDotProductAttention(queries, keys, values, mask *Array, scale float32, s *Stream) *Array {
	r := &Array{}
	var maskArr C.mlx_array
	modeStr := "none"
	if mask != nil {
		maskArr = mask.handle
		modeStr = "additive"
	}
	maskMode := C.CString(modeStr)
	defer C.free(unsafe.Pointer(maskMode))
	var sinks C.mlx_array // null
	rc := C.mlx_fast_scaled_dot_product_attention(&r.handle,
		queries.handle, keys.handle, values.handle,
		C.float(scale), maskMode, maskArr, sinks, s.handle)
	checkRC(rc, "mlx_fast_scaled_dot_product_attention")
	return r
}

// ===========================================================================
// Quantization
// ===========================================================================

// Quantize quantizes an array.
func Quantize(w *Array, groupSize, bits int, s *Stream) (quantized, scales, biases *Array) {
	var vecRes C.mlx_vector_array
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_quantize(&vecRes, w.handle, optGroupSize, optBits, cMode, s.handle)
	checkRC(rc, "mlx_quantize")
	v := &VectorArray{handle: vecRes}
	defer v.Free()
	quantized = v.Get(0)
	scales = v.Get(1)
	biases = v.Get(2)
	return
}

// Dequantize dequantizes an array.
func Dequantize(w, scalesArr, biasesArr *Array, groupSize, bits int, s *Stream) *Array {
	r := &Array{}
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	optDtype := C.mlx_optional_dtype{has_value: false}
	rc := C.mlx_dequantize(&r.handle, w.handle, scalesArr.handle, biasesArr.handle,
		optGroupSize, optBits, cMode, optDtype, s.handle)
	checkRC(rc, "mlx_dequantize")
	return r
}

// QuantizedMatMul performs quantized matrix multiplication.
func QuantizedMatMul(x, w, scalesArr, biasesArr *Array, transpose bool, groupSize, bits int, s *Stream) *Array {
	r := &Array{}
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_quantized_matmul(&r.handle, x.handle, w.handle, scalesArr.handle, biasesArr.handle,
		C.bool(transpose), optGroupSize, optBits, cMode, s.handle)
	checkRC(rc, "mlx_quantized_matmul")
	return r
}

// ===========================================================================
// IsInf (for IsFinite implementation)
// ===========================================================================

// IsInf checks for infinity.
func IsInf(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_isinf(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_isinf")
	return r
}

// Full creates an array filled with a scalar value.
func Full(shape []int, val *Array, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_full(&r.handle, shapePtr, C.size_t(len(shape)), val.handle, dtype, s.handle)
	checkRC(rc, "mlx_full")
	return r
}

// Zeros creates a zero-filled array.
func Zeros(shape []int, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_zeros(&r.handle, shapePtr, C.size_t(len(shape)), dtype, s.handle)
	checkRC(rc, "mlx_zeros")
	return r
}

// Ones creates a ones-filled array.
func Ones(shape []int, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_ones(&r.handle, shapePtr, C.size_t(len(shape)), dtype, s.handle)
	checkRC(rc, "mlx_ones")
	return r
}

// StopGradient prevents gradient flow through this array.
func StopGradient(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_stop_gradient(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_stop_gradient")
	return r
}

// Copy creates a copy of the array.
func Copy(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_copy(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_copy")
	return r
}

// AsStrided creates a view of the array with custom strides and shape.
func AsStrided(x *Array, shape []int, strides []int64, offset int, s *Stream) *Array {
	r := &Array{}
	var shapePtr *C.int
	var stridesPtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int)(unsafe.Pointer(&shape[0]))
	}
	if len(strides) > 0 {
		stridesPtr = (*C.int64_t)(unsafe.Pointer(&strides[0]))
	}
	rc := C.mlx_as_strided(&r.handle, x.handle,
		shapePtr, C.size_t(len(shape)),
		stridesPtr, C.size_t(len(strides)),
		C.size_t(offset), s.handle)
	checkRC(rc, "mlx_as_strided")
	return r
}

// ArraySet replaces the contents of dst with src, preserving the dst handle.
func ArraySet(dst, src *Array) {
	rc := C.mlx_array_set(&dst.handle, src.handle)
	checkRC(rc, "mlx_array_set")
}

// ===========================================================================
// Closure (for compiled function execution)
// ===========================================================================

// GoClosureFunc is the Go function type that closures wrap.
// It takes input arrays and returns output arrays.
type GoClosureFunc func(inputs []*Array) []*Array

// closureRegistry maps payload IDs to Go closure functions.
var closureRegistry struct {
	sync.Mutex
	funcs   map[uintptr]GoClosureFunc
	counter uintptr
}

func init() {
	closureRegistry.funcs = make(map[uintptr]GoClosureFunc)
}

//export goClosureCallback
func goClosureCallback(res *C.mlx_vector_array, inputs C.mlx_vector_array, payload unsafe.Pointer) C.int {
	id := uintptr(payload)
	closureRegistry.Lock()
	fn, ok := closureRegistry.funcs[id]
	closureRegistry.Unlock()
	if !ok {
		return 1 // error: callback not found
	}

	// Unpack input arrays from the vector.
	inVec := &VectorArray{handle: inputs}
	n := inVec.Size()
	goInputs := make([]*Array, n)
	for i := 0; i < n; i++ {
		goInputs[i] = inVec.Get(i)
	}

	// Call the Go function.
	goOutputs := fn(goInputs)

	// Pack output arrays into the result vector.
	outVec := NewVectorArray(goOutputs)
	*res = outVec.handle

	return 0
}

// Closure wraps an mlx_closure that maps inputs to outputs.
type Closure struct {
	handle     C.mlx_closure
	registryID uintptr // non-zero if registered in closureRegistry
}

// NewClosureFromGoFunc creates an MLX closure backed by a Go function.
func NewClosureFromGoFunc(fn GoClosureFunc) *Closure {
	closureRegistry.Lock()
	closureRegistry.counter++
	id := closureRegistry.counter
	closureRegistry.funcs[id] = fn
	closureRegistry.Unlock()

	cl := &Closure{
		registryID: id,
	}
	cl.handle = C.create_go_closure(unsafe.Pointer(id))
	return cl
}

// CompileClosure wraps a closure with mlx_compile for fused GPU execution.
func CompileClosure(cl *Closure, shapeless bool) *Closure {
	compiled := &Closure{}
	rc := C.mlx_compile(&compiled.handle, cl.handle, C.bool(shapeless))
	checkRC(rc, "mlx_compile")
	return compiled
}

// Free releases the closure and unregisters from the registry if needed.
func (cl *Closure) Free() {
	if cl.registryID != 0 {
		closureRegistry.Lock()
		delete(closureRegistry.funcs, cl.registryID)
		closureRegistry.Unlock()
		cl.registryID = 0
	}
	C.mlx_closure_free(cl.handle)
}

// ApplyClosure calls the closure with input arrays and returns output arrays.
func ApplyClosure(cl *Closure, inputs []*Array) ([]*Array, error) {
	inVec := NewVectorArray(inputs)
	defer inVec.Free()
	var outVec C.mlx_vector_array
	rc := C.mlx_closure_apply(&outVec, cl.handle, inVec.handle)
	if err := checkRCErr(rc, "mlx_closure_apply"); err != nil {
		return nil, err
	}
	v := &VectorArray{handle: outVec}
	defer v.Free()
	n := v.Size()
	results := make([]*Array, n)
	for i := 0; i < n; i++ {
		results[i] = v.Get(i)
	}
	return results, nil
}
