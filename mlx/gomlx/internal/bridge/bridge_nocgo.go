// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !darwin

// Package bridge provides CGo bindings to Apple's MLX framework.
// This file is a stub for when CGO is disabled or on non-Darwin platforms.
package bridge

import (
	"fmt"
	"unsafe"
)

type DType = int32

var (
	DTypeBool      DType = 0
	DTypeUint8     DType = 1
	DTypeUint16    DType = 2
	DTypeUint32    DType = 3
	DTypeUint64    DType = 4
	DTypeInt8      DType = 5
	DTypeInt16     DType = 6
	DTypeInt32     DType = 7
	DTypeInt64     DType = 8
	DTypeFloat16   DType = 9
	DTypeBFloat16  DType = 10
	DTypeFloat32   DType = 11
	DTypeFloat64   DType = 12
	DTypeComplex64 DType = 13
)

const (
	ReduceSum     = 0
	ReduceProduct = 1
	ReduceMax     = 2
	ReduceMin     = 3
)

const (
	ScatterModeAdd = 0
	ScatterModeMax = 1
	ScatterModeMin = 2
)

var errNoCGo = fmt.Errorf("mlx: CGO is disabled or platform is not Darwin; rebuild with CGO_ENABLED=1 on macOS")

type Array struct{}
type VectorArray struct{}
type Stream struct{}

func NewArray() *Array                                                              { return nil }
func NewArrayFromData(data unsafe.Pointer, shape []int, dtype DType) *Array         { return nil }
func NewArrayScalarFloat32(val float32) *Array                                      { return nil }
func NewArrayScalarInt32(val int32) *Array                                          { return nil }
func NewArrayScalarBool(val bool) *Array                                            { return nil }
func (a *Array) Free()                                                              {}
func (a *Array) Shape() []int                                                       { return nil }
func (a *Array) NDim() int                                                          { return 0 }
func (a *Array) Size() int                                                          { return 0 }
func (a *Array) NBytes() int                                                        { return 0 }
func (a *Array) DType() DType                                                       { return 0 }
func (a *Array) ItemSize() int                                                      { return 0 }
func (a *Array) DataPtr() unsafe.Pointer                                            { return nil }
func NewVectorArray(arrays []*Array) *VectorArray                                   { return nil }
func (v *VectorArray) Free()                                                        {}
func (v *VectorArray) Size() int                                                    { return 0 }
func (v *VectorArray) Get(index int) *Array                                         { return nil }
func DefaultGPUStream() *Stream                                                     { return nil }
func DefaultCPUStream() *Stream                                                     { return nil }
func (s *Stream) Free()                                                             {}
func Eval(arrays ...*Array) error                                                   { return errNoCGo }
func MetalIsAvailable() bool                                                        { return false }
func ClearCache()                                                                   {}
func GetActiveMemory() uint64                                                       { return 0 }
func SetMemoryLimit(limit uint64) uint64                                            { return 0 }
func SetCacheLimit(limit uint64) uint64                                             { return 0 }

func Abs(x *Array, s *Stream) *Array                                       { return nil }
func Negative(x *Array, s *Stream) *Array                                  { return nil }
func Sqrt(x *Array, s *Stream) *Array                                      { return nil }
func Rsqrt(x *Array, s *Stream) *Array                                     { return nil }
func Exp(x *Array, s *Stream) *Array                                       { return nil }
func Expm1(x *Array, s *Stream) *Array                                     { return nil }
func Log(x *Array, s *Stream) *Array                                       { return nil }
func Log1p(x *Array, s *Stream) *Array                                     { return nil }
func Sin(x *Array, s *Stream) *Array                                       { return nil }
func Cos(x *Array, s *Stream) *Array                                       { return nil }
func Tanh(x *Array, s *Stream) *Array                                      { return nil }
func Sigmoid(x *Array, s *Stream) *Array                                   { return nil }
func Erf(x *Array, s *Stream) *Array                                       { return nil }
func Floor(x *Array, s *Stream) *Array                                     { return nil }
func Ceil(x *Array, s *Stream) *Array                                      { return nil }
func Round(x *Array, s *Stream) *Array                                     { return nil }
func Sign(x *Array, s *Stream) *Array                                      { return nil }
func LogicalNot(x *Array, s *Stream) *Array                                { return nil }
func IsNaN(x *Array, s *Stream) *Array                                     { return nil }
func BitwiseNot(x *Array, s *Stream) *Array                                { return nil }

func Add(lhs, rhs *Array, s *Stream) *Array                               { return nil }
func Subtract(lhs, rhs *Array, s *Stream) *Array                          { return nil }
func Multiply(lhs, rhs *Array, s *Stream) *Array                          { return nil }
func Divide(lhs, rhs *Array, s *Stream) *Array                            { return nil }
func Remainder(lhs, rhs *Array, s *Stream) *Array                         { return nil }
func Power(lhs, rhs *Array, s *Stream) *Array                             { return nil }
func Maximum(lhs, rhs *Array, s *Stream) *Array                           { return nil }
func Minimum(lhs, rhs *Array, s *Stream) *Array                           { return nil }
func Arctan2(lhs, rhs *Array, s *Stream) *Array                           { return nil }
func LogicalAnd(lhs, rhs *Array, s *Stream) *Array                        { return nil }
func LogicalOr(lhs, rhs *Array, s *Stream) *Array                         { return nil }
func BitwiseAnd(lhs, rhs *Array, s *Stream) *Array                        { return nil }
func BitwiseOr(lhs, rhs *Array, s *Stream) *Array                         { return nil }
func BitwiseXor(lhs, rhs *Array, s *Stream) *Array                        { return nil }
func LeftShift(lhs, rhs *Array, s *Stream) *Array                         { return nil }
func RightShift(lhs, rhs *Array, s *Stream) *Array                        { return nil }
func Equal(lhs, rhs *Array, s *Stream) *Array                             { return nil }
func NotEqual(lhs, rhs *Array, s *Stream) *Array                          { return nil }
func Less(lhs, rhs *Array, s *Stream) *Array                              { return nil }
func LessEqual(lhs, rhs *Array, s *Stream) *Array                         { return nil }
func Greater(lhs, rhs *Array, s *Stream) *Array                           { return nil }
func GreaterEqual(lhs, rhs *Array, s *Stream) *Array                      { return nil }

func Reshape(x *Array, shape []int, s *Stream) *Array                     { return nil }
func Transpose(x *Array, axes []int, s *Stream) *Array                    { return nil }
func BroadcastTo(x *Array, shape []int, s *Stream) *Array                 { return nil }
func AsType(x *Array, dtype DType, s *Stream) *Array                      { return nil }
func Concatenate(arrays []*Array, axis int, s *Stream) *Array             { return nil }
func Slice(x *Array, starts, stops, strides []int, s *Stream) *Array      { return nil }
func SliceUpdate(src, update *Array, starts, stops, strides []int, s *Stream) *Array { return nil }
func Pad(x, padValue *Array, axes []int, lowPads, highPads []int, s *Stream) *Array  { return nil }
func Flip(x *Array, axes []int, s *Stream) *Array                         { return nil }
func ExpandDims(x *Array, axes []int, s *Stream) *Array                   { return nil }
func Squeeze(x *Array, axes []int, s *Stream) *Array                      { return nil }
func Where(condition, x, y *Array, s *Stream) *Array                      { return nil }
func Clip(x, min, max *Array, s *Stream) *Array                           { return nil }

func Sum(x *Array, axes []int, keepDims bool, s *Stream) *Array           { return nil }
func Prod(x *Array, axes []int, keepDims bool, s *Stream) *Array          { return nil }
func Max(x *Array, axes []int, keepDims bool, s *Stream) *Array           { return nil }
func Min(x *Array, axes []int, keepDims bool, s *Stream) *Array           { return nil }
func All(x *Array, axes []int, keepDims bool, s *Stream) *Array           { return nil }
func Any(x *Array, axes []int, keepDims bool, s *Stream) *Array           { return nil }
func ArgMin(x *Array, axis int, keepDims bool, s *Stream) *Array          { return nil }
func ArgMax(x *Array, axis int, keepDims bool, s *Stream) *Array          { return nil }

func MatMul(lhs, rhs *Array, s *Stream) *Array                            { return nil }
func Take(x, indices *Array, axis int, s *Stream) *Array                   { return nil }
func TakeAlongAxis(x, indices *Array, axis int, s *Stream) *Array          { return nil }
func Gather(x *Array, indices []*Array, axes []int, sliceSizes []int, s *Stream) *Array { return nil }
func ScatterAdd(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array { return nil }
func ScatterMax(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array { return nil }
func ScatterMin(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array { return nil }

func Conv2d(input, weight *Array, stride, padding, dilation [2]int, groups int, s *Stream) *Array { return nil }
func Conv1d(input, weight *Array, stride, padding, dilation, groups int, s *Stream) *Array         { return nil }
func Softmax(x *Array, axis int, precise bool, s *Stream) *Array                                   { return nil }
func Sort(x *Array, axis int, s *Stream) *Array                                                    { return nil }
func ArgSort(x *Array, axis int, s *Stream) *Array                                                 { return nil }
func Arange(start, stop, step float64, dtype DType, s *Stream) *Array                             { return nil }

func RandomKey(seed uint64) *Array                                                                  { return nil }
func RandomSplit(key *Array, s *Stream) (*Array, *Array)                                           { return nil, nil }
func RandomUniform(low, high *Array, shape []int, dtype DType, key *Array, s *Stream) *Array       { return nil }
func RandomNormal(shape []int, dtype DType, key *Array, s *Stream) *Array                          { return nil }
func RandomBits(shape []int, width int, key *Array, s *Stream) *Array                              { return nil }

func FastLayerNorm(x, weight, bias *Array, eps float32, s *Stream) *Array                          { return nil }
func FastRMSNorm(x, weight *Array, eps float32, s *Stream) *Array                                  { return nil }
func FastRope(x *Array, dims int, traditional bool, base, scale float32, offset int, s *Stream) *Array { return nil }
func FastScaledDotProductAttention(q, k, v, mask *Array, scale float32, s *Stream) *Array           { return nil }

func Quantize(w *Array, groupSize, bits int, s *Stream) (quantized, scales, biases *Array) { return nil, nil, nil }
func Dequantize(w, scales, biases *Array, groupSize, bits int, s *Stream) *Array            { return nil }
func QuantizedMatMul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, s *Stream) *Array { return nil }

func IsInf(x *Array, s *Stream) *Array                                                             { return nil }
func Full(shape []int, val *Array, dtype DType, s *Stream) *Array                                  { return nil }
func Zeros(shape []int, dtype DType, s *Stream) *Array                                             { return nil }
func Ones(shape []int, dtype DType, s *Stream) *Array                                              { return nil }
func StopGradient(x *Array, s *Stream) *Array                                                      { return nil }
func Copy(x *Array, s *Stream) *Array                                                              { return nil }
func ArraySet(dst, src *Array)                                                                      {}
func AsStrided(x *Array, shape []int, strides []int64, offset int, s *Stream) *Array               { return nil }

type Closure struct{}

func NewClosure(inputs, outputs []*Array) *Closure                         { return nil }
func (c *Closure) Free()                                                    {}
func ApplyClosure(cl *Closure, inputs []*Array) ([]*Array, error)          { return nil, errNoCGo }
