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

type GoClosureFunc func(inputs []*Array) []*Array

type Closure struct{}

func NewClosureFromGoFunc(fn GoClosureFunc) *Closure                       { return nil }
func CompileClosure(cl *Closure, shapeless bool) *Closure                  { return nil }
func (c *Closure) Free()                                                    {}
func ApplyClosure(cl *Closure, inputs []*Array) ([]*Array, error)          { return nil, errNoCGo }
func NewClosureFromCTape(instrs []int32, numSlots int, outputSlots, constSlots []int32, constArrays []*Array, paramSlots []int32) *Closure {
	return nil
}

// Autograd transforms stubs.

type ValueAndGradClosure struct{}

func ValueAndGrad(fn *Closure, argnums []int) *ValueAndGradClosure     { return nil }
func (vg *ValueAndGradClosure) Apply(inputs []*Array) ([]*Array, []*Array, error) {
	return nil, nil, errNoCGo
}
func (vg *ValueAndGradClosure) Free()                                    {}
func VJP(fn *Closure, primals []*Array, cotangents []*Array) ([]*Array, []*Array, error) {
	return nil, nil, errNoCGo
}
func JVP(fn *Closure, primals []*Array, tangents []*Array) ([]*Array, []*Array, error) {
	return nil, nil, errNoCGo
}
func Checkpoint(fn *Closure) *Closure                                    { return nil }
func AsyncEval(arrays ...*Array) error                                   { return errNoCGo }

// Memory profiling stubs.
func GetCacheMemory() uint64                                             { return 0 }
func GetPeakMemory() uint64                                              { return 0 }
func ResetPeakMemory()                                                   {}
func SetWiredLimit(limit uint64) uint64                                  { return 0 }

// Cumulative ops stubs.
func CumSum(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array  { return nil }
func CumProd(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array { return nil }
func CumMax(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array  { return nil }
func CumMin(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array  { return nil }

// Statistical ops stubs.
func Mean(x *Array, keepDims bool, s *Stream) *Array                                 { return nil }
func MeanAxes(x *Array, axes []int, keepDims bool, s *Stream) *Array                 { return nil }
func MeanAxis(x *Array, axis int, keepDims bool, s *Stream) *Array                   { return nil }
func Variance(x *Array, keepDims bool, ddof int, s *Stream) *Array                   { return nil }
func VarianceAxes(x *Array, axes []int, keepDims bool, ddof int, s *Stream) *Array   { return nil }
func StdDev(x *Array, keepDims bool, ddof int, s *Stream) *Array                     { return nil }
func StdDevAxes(x *Array, axes []int, keepDims bool, ddof int, s *Stream) *Array     { return nil }
func LogSumExp(x *Array, keepDims bool, s *Stream) *Array                             { return nil }
func LogSumExpAxes(x *Array, axes []int, keepDims bool, s *Stream) *Array             { return nil }
func LogAddExp(a, b *Array, s *Stream) *Array                                         { return nil }

// Additional array ops stubs.
func TopK(x *Array, k int, s *Stream) *Array                                          { return nil }
func TopKAxis(x *Array, k, axis int, s *Stream) *Array                                { return nil }
func Flatten(x *Array, startAxis, endAxis int, s *Stream) *Array                      { return nil }
func Unflatten(x *Array, axis int, shape []int, s *Stream) *Array                     { return nil }
func Diagonal(x *Array, offset, axis1, axis2 int, s *Stream) *Array                   { return nil }
func Trace(x *Array, offset, axis1, axis2 int, dtype DType, s *Stream) *Array         { return nil }
func Tile(x *Array, reps []int, s *Stream) *Array                                     { return nil }
func Repeat(x *Array, repeats int, s *Stream) *Array                                  { return nil }
func RepeatAxis(x *Array, repeats, axis int, s *Stream) *Array                        { return nil }
func Split(x *Array, numSplits, axis int, s *Stream) *VectorArray                     { return nil }
func SplitSections(x *Array, indices []int, axis int, s *Stream) *VectorArray         { return nil }
func Linspace(start, stop float64, num int, dtype DType, s *Stream) *Array            { return nil }
func AddMM(c, a, b *Array, alpha, beta float32, s *Stream) *Array                     { return nil }

// Advanced RNG stubs.
func RandomBernoulli(p *Array, shape []int, key *Array, s *Stream) *Array                          { return nil }
func RandomCategorical(logits *Array, axis int, key *Array, s *Stream) *Array                      { return nil }
func RandomCategoricalNumSamples(logits *Array, axis, numSamples int, key *Array, s *Stream) *Array { return nil }
func RandomPermutation(x *Array, axis int, key *Array, s *Stream) *Array                           { return nil }
func RandomPermutationArange(n int, key *Array, s *Stream) *Array                                  { return nil }
func RandomRandInt(low, high *Array, shape []int, dtype DType, key *Array, s *Stream) *Array        { return nil }
func RandomGumbel(shape []int, dtype DType, key *Array, s *Stream) *Array                          { return nil }
func RandomLaplace(shape []int, dtype DType, loc, scale float32, key *Array, s *Stream) *Array     { return nil }
func RandomTruncatedNormal(lower, upper *Array, shape []int, dtype DType, key *Array, s *Stream) *Array { return nil }
func RandomSeed(seed uint64)                                                                        {}

// Linear algebra stubs.
func LinalgCholesky(a *Array, upper bool, s *Stream) *Array                            { return nil }
func LinalgInv(a *Array, s *Stream) *Array                                             { return nil }
func LinalgSolve(a, b *Array, s *Stream) *Array                                        { return nil }
func LinalgSolveTriangular(a, b *Array, upper bool, s *Stream) *Array                  { return nil }
func LinalgSVD(a *Array, computeUV bool, s *Stream) *VectorArray                       { return nil }
func LinalgQR(a *Array, s *Stream) (*Array, *Array)                                    { return nil, nil }
func LinalgEig(a *Array, s *Stream) (*Array, *Array)                                   { return nil, nil }
func LinalgEigh(a *Array, uplo string, s *Stream) (*Array, *Array)                     { return nil, nil }
func LinalgEigvals(a *Array, s *Stream) *Array                                         { return nil }
func LinalgNorm(a *Array, ord float64, axes []int, keepDims bool, s *Stream) *Array    { return nil }
func LinalgNormL2(a *Array, axes []int, keepDims bool, s *Stream) *Array               { return nil }
func LinalgPinv(a *Array, s *Stream) *Array                                            { return nil }
func LinalgLU(a *Array, s *Stream) *VectorArray                                        { return nil }
func LinalgLUFactor(a *Array, s *Stream) (*Array, *Array)                              { return nil, nil }
func LinalgCross(a, b *Array, axis int, s *Stream) *Array                              { return nil }

// Transpose convolution stubs.
func ConvTranspose1d(input, weight *Array, stride, padding, dilation, outputPadding, groups int, s *Stream) *Array { return nil }
func ConvTranspose2d(input, weight *Array, stride, padding, dilation, outputPadding [2]int, groups int, s *Stream) *Array { return nil }
func ConvTranspose3d(input, weight *Array, stride, padding, dilation, outputPadding [3]int, groups int, s *Stream) *Array { return nil }

// SafeTensors IO stubs.
type MapStringToArray struct{}
type MapStringToString struct{}

func NewMapStringToArray() *MapStringToArray    { return nil }
func (m *MapStringToArray) Free()               {}
func (m *MapStringToArray) Insert(key string, value *Array) {}
func (m *MapStringToArray) Get(key string) *Array { return nil }
func (m *MapStringToArray) Iterate(fn func(key string, value *Array)) {}

func NewMapStringToString() *MapStringToString  { return nil }
func (m *MapStringToString) Free()              {}
func (m *MapStringToString) Insert(key, value string) {}
func (m *MapStringToString) Get(key string) (string, bool) { return "", false }
func (m *MapStringToString) Iterate(fn func(key, value string)) {}

func LoadSafeTensors(path string, s *Stream) (*MapStringToArray, *MapStringToString, error) { return nil, nil, errNoCGo }
func SaveSafeTensors(path string, arrays *MapStringToArray, metadata *MapStringToString) error { return errNoCGo }
func LoadArray(path string, s *Stream) (*Array, error) { return nil, errNoCGo }
func SaveArray(path string, a *Array) error { return errNoCGo }
