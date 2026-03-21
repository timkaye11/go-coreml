// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"
	"reflect"
	"runtime"
	"slices"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// ===========================================================================
// C Tape Opcodes — must match enum tape_opcode in bridge.go's C preamble.
// ===========================================================================

const (
	// Unary
	opAbs int32 = iota
	opNeg
	opSqrt
	opRsqrt
	opExp
	opExpm1
	opLog
	opLog1p
	opSin
	opCos
	opTanh
	opSigmoid
	opErf
	opFloor
	opCeil
	opRound
	opSign
	opLogicalNot
	opBitwiseNot
	opIsNaN
	opIsInf
	// Binary
	opAdd
	opSub
	opMul
	opDiv
	opRem
	opPow
	opMax
	opMin
	opAtan2
	opLogicalAnd
	opLogicalOr
	opBitwiseAnd
	opBitwiseOr
	opBitwiseXor
	opLeftShift
	opRightShift
	opEq
	opNe
	opLt
	opLe
	opGt
	opGe
	opMatMul
	// Ternary
	opWhere
	opClip
	// Shape
	opReshape
	opTranspose
	opBroadcastTo
	opSlice
	opSqueeze
	opExpandDims
	opFlip
	// Reduce
	opReduceSum
	opReduceMax
	opReduceMin
	opReduceProd
	opReduceAll
	opReduceAny
	opArgMin
	opArgMax
	// Type
	opAsType
	// Special
	opSoftmax
	opPad
	opConcat
	opTake
	opTakeAlongAxis
	opConv1d
	opConv2d
	opScatterAdd
	opScatterMax
	opScatterMin
	opFastLayerNorm
	opFastRMSNorm
	opFastSDPA
	opFastRope
	opSliceUpdate
	// Scalar creation
	opNewScalarF32
	opNewScalarI32
	opNewScalarBool
	// Temp management
	opFreeTemp
	// Misc
	opStopGradient
	opCopy
	opCount // sentinel
)

// instrMeta captures operation metadata for the C tape interpreter.
type instrMeta struct {
	opcode  int32
	outSlot int32   // tape index for main entries; > len(tape) for temps
	inputs  []int32 // tape indices of inputs
	params  []int32 // shapes, axes, dtypes-as-ints, floats-as-bits
}

// float32Bits converts a float32 to its int32 bit representation for params.
func float32Bits(f float32) int32 {
	return int32(math.Float32bits(f))
}

// graphNode represents a value in the computation graph.
type graphNode struct {
	array   *bridge.Array // MLX array (lazy — not evaluated until needed)
	shape   shapes.Shape
	name    string    // Optional name (for parameters)
	owner   *Function // Which function created this node
	tapeIdx int       // Index into owner's tape
}

// Function implements backends.Function for the MLX backend.
type Function struct {
	notimplemented.Function // Bootstrap: unimplemented ops return ErrNotImplemented.

	builder  *Builder
	name     string
	parent   *Function
	returned bool
	params   []*graphNode // Input placeholders
	outputs  []*graphNode // Return values

	// Tape records all graph nodes for replay during Execute.
	// Each entry is a closure that, given the arrays produced so far and a stream,
	// returns a new array. Parameters have nil closures (substituted from inputs).
	tape         []func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array
	constIndices []int // Tape indices of constants (must not be freed during replay cleanup)

	// buildArrays tracks all MLX arrays created during graph building,
	// so they can be freed when the function is finalized.
	buildArrays []*bridge.Array

	// controlFlowStep records a control flow operation (While/If/Sort/Call).
	controlFlowStep *controlFlowStep

	// C tape interpreter metadata — parallel to tape entries.
	instrs        []instrMeta // instruction metadata for C interpreter
	hasGoCallback bool        // true if any op needs Go callback (DynamicSlice, RNG)
	tempSlotCount int         // number of temp slots for compound ops
	goCallbackOps map[string]struct{}
}

// controlFlowStep records a pending control flow operation.
type controlFlowStep struct {
	opType       backends.OpType
	inputs       []*graphNode
	outputShapes []shapes.Shape
	outputNodes  []*graphNode
	whileData    *whileStepData
	ifData       *ifStepData
	sortData     *sortStepData
	callData     *callStepData
}

type whileStepData struct {
	condFn *Function
	bodyFn *Function
}
type ifStepData struct {
	trueFn  *Function
	falseFn *Function
}
type sortStepData struct {
	comparatorFn *Function
	axis         int
	isStable     bool
}
type callStepData struct {
	targetFn *Function
}

var _ backends.Function = &Function{}

func newFunction(builder *Builder, name string, parent *Function) *Function {
	return &Function{
		builder: builder,
		name:    name,
		parent:  parent,
	}
}

func (f *Function) Name() string { return f.name }

func (f *Function) Parent() backends.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// Closure creates a new function for control flow bodies.
func (f *Function) Closure() (backends.Function, error) {
	return newFunction(f.builder, "", f), nil
}

// stream returns the backend's GPU stream.
func (f *Function) stream() *bridge.Stream {
	return f.builder.backend.stream()
}

// resolveNode converts a backends.Value to a graphNode.
func (f *Function) resolveNode(v backends.Value) (*graphNode, error) {
	node, ok := v.(*graphNode)
	if !ok {
		return nil, errors.Errorf("expected *graphNode, got %T", v)
	}
	return node, nil
}

// resolveNodes converts multiple backends.Value to graphNodes.
func (f *Function) resolveNodes(name string, values ...backends.Value) ([]*graphNode, error) {
	nodes := make([]*graphNode, len(values))
	for i, v := range values {
		n, err := f.resolveNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: input #%d", name, i)
		}
		nodes[i] = n
	}
	return nodes, nil
}

// validateClosure validates that a backends.Function is a compiled closure.
func (f *Function) validateClosure(opName, closureName string, closure backends.Function) (*Function, error) {
	fn, ok := closure.(*Function)
	if !ok {
		return nil, errors.Errorf("%s: %s must be a *mlx.Function, got %T", opName, closureName, closure)
	}
	if fn.parent != f {
		return nil, errors.Errorf("%s: %s must be a closure of the current function", opName, closureName)
	}
	if !fn.returned {
		return nil, errors.Errorf("%s: %s must have Return() called", opName, closureName)
	}
	return fn, nil
}

// record appends a tape entry and returns the new graphNode.
// The tapeFn closure takes (all arrays produced so far, stream) and returns a new array.
func (f *Function) record(shape shapes.Shape, arr *bridge.Array, tapeFn func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array) *graphNode {
	idx := len(f.tape)
	f.tape = append(f.tape, tapeFn)
	if arr != nil {
		f.buildArrays = append(f.buildArrays, arr)
	}
	return &graphNode{array: arr, shape: shape, owner: f, tapeIdx: idx}
}

// emitInstr appends an instruction metadata entry for the C tape interpreter.
func (f *Function) emitInstr(opcode int32, outSlot int32, inputs []int32, params ...int32) {
	f.instrs = append(f.instrs, instrMeta{
		opcode:  opcode,
		outSlot: outSlot,
		inputs:  inputs,
		params:  params,
	})
}

// tempSlotBase is a sentinel offset used for temp slots during graph building.
// Temp slots are stored as tempSlotBase+N during building, then remapped to
// len(f.tape)+N during serialization by serializeTape().
const tempSlotBase = 1 << 20 // 1M — well above any realistic tape length

// nextTempSlot allocates a temporary slot for compound C tape ops.
// Returns a sentinel index (tempSlotBase + N) that gets remapped to the correct
// range [len(f.tape), len(f.tape)+tempSlotCount) during serializeTape().
func (f *Function) nextTempSlot() int32 {
	slot := int32(tempSlotBase) + int32(f.tempSlotCount)
	f.tempSlotCount++
	return slot
}

// markGoCallback marks this function as requiring a Go callback (fallback path).
func (f *Function) markGoCallback(op string) {
	f.hasGoCallback = true
	if op == "" {
		return
	}
	if f.goCallbackOps == nil {
		f.goCallbackOps = make(map[string]struct{})
	}
	f.goCallbackOps[op] = struct{}{}
}

func (f *Function) goCallbackReasons() []string {
	if len(f.goCallbackOps) == 0 {
		return nil
	}
	reasons := make([]string, 0, len(f.goCallbackOps))
	for op := range f.goCallbackOps {
		reasons = append(reasons, op)
	}
	slices.Sort(reasons)
	return reasons
}

// finalize frees all build-time MLX arrays held by this function.
func (f *Function) finalize() {
	for _, arr := range f.buildArrays {
		if arr != nil {
			arr.Free()
		}
	}
	f.buildArrays = nil
}

// makeScalarConst creates a scalar constant of the given dtype.
func (f *Function) makeScalarConst(val float64, dt dtypes.DType) *bridge.Array {
	arr := bridge.NewArrayScalarFloat32(float32(val))
	if dt != dtypes.Float32 {
		casted := bridge.AsType(arr, gomlxDTypeToMLX(dt), f.stream())
		arr.Free()
		return casted
	}
	return arr
}

// ===========================================================================
// Lifecycle: Parameter, Constant, Return
// ===========================================================================

// Parameter creates an input placeholder.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Value, error) {
	dims := shape.Dimensions
	mlxDType := gomlxDTypeToMLX(shape.DType)
	arr := bridge.Zeros(dims, mlxDType, f.stream())
	idx := len(f.tape)
	// Parameter tape entries have nil closure — they are substituted during Execute.
	f.tape = append(f.tape, nil)
	node := &graphNode{array: arr, shape: shape, name: name, owner: f, tapeIdx: idx}
	f.params = append(f.params, node)
	f.buildArrays = append(f.buildArrays, arr)
	return node, nil
}

// Constant creates a constant tensor.
func (f *Function) Constant(flat any, dims ...int) (backends.Value, error) {
	flatVal := reflect.ValueOf(flat)
	if flatVal.Kind() != reflect.Slice {
		return nil, errors.Errorf("Constant: expected slice, got %T", flat)
	}
	dt := dtypes.FromGoType(flatVal.Type().Elem())
	shape := shapes.Make(dt, dims...)

	var dataPtr unsafe.Pointer
	if flatVal.Len() > 0 {
		dataPtr = unsafe.Pointer(flatVal.Pointer())
	}

	mlxDType := gomlxDTypeToMLX(dt)
	arr := bridge.NewArrayFromData(dataPtr, dims, mlxDType)
	runtime.KeepAlive(flat)

	constArr := arr
	node := f.record(shape, arr, func(_ []*bridge.Array, _ *bridge.Stream) *bridge.Array {
		return constArr
	})
	f.constIndices = append(f.constIndices, node.tapeIdx)
	return node, nil
}

// Return marks the function outputs.
func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	nodes, err := f.resolveNodes("Return", outputs...)
	if err != nil {
		return err
	}
	f.outputs = nodes
	f.returned = true
	return nil
}

// ===========================================================================
// Unary Operations
// ===========================================================================

// unaryOp creates a unary operation, records it on the tape, and returns the result.
func (f *Function) unaryOp(bridgeFn func(*bridge.Array, *bridge.Stream) *bridge.Array, x *graphNode, opcodes ...int32) *graphNode {
	r := bridgeFn(x.array, f.stream())
	xi := x.tapeIdx
	node := f.record(x.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridgeFn(arrays[xi], s)
	})
	if len(opcodes) > 0 {
		f.emitInstr(opcodes[0], int32(node.tapeIdx), []int32{int32(xi)})
	}
	return node
}

// unaryBoolOp is like unaryOp but the output has dtype Bool regardless of the input dtype.
func (f *Function) unaryBoolOp(bridgeFn func(*bridge.Array, *bridge.Stream) *bridge.Array, x *graphNode, opcodes ...int32) *graphNode {
	r := bridgeFn(x.array, f.stream())
	outShape := x.shape.Clone()
	outShape.DType = dtypes.Bool
	xi := x.tapeIdx
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridgeFn(arrays[xi], s)
	})
	if len(opcodes) > 0 {
		f.emitInstr(opcodes[0], int32(node.tapeIdx), []int32{int32(xi)})
	}
	return node
}

func (f *Function) Abs(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Abs, n, opAbs), nil
}
func (f *Function) Neg(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Negative, n, opNeg), nil
}
func (f *Function) Sqrt(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Sqrt, n, opSqrt), nil
}
func (f *Function) Rsqrt(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Rsqrt, n, opRsqrt), nil
}
func (f *Function) Exp(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Exp, n, opExp), nil
}
func (f *Function) Expm1(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Expm1, n, opExpm1), nil
}
func (f *Function) Log(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Log, n, opLog), nil
}
func (f *Function) Log1p(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Log1p, n, opLog1p), nil
}
func (f *Function) Sin(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Sin, n, opSin), nil
}
func (f *Function) Cos(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Cos, n, opCos), nil
}
func (f *Function) Tanh(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Tanh, n, opTanh), nil
}
func (f *Function) Logistic(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Sigmoid, n, opSigmoid), nil
}
func (f *Function) Erf(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Erf, n, opErf), nil
}
func (f *Function) Floor(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Floor, n, opFloor), nil
}
func (f *Function) Ceil(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Ceil, n, opCeil), nil
}
func (f *Function) Round(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Round, n, opRound), nil
}
func (f *Function) Sign(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.Sign, n, opSign), nil
}
func (f *Function) LogicalNot(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryBoolOp(bridge.LogicalNot, n, opLogicalNot), nil
}
func (f *Function) BitwiseNot(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryOp(bridge.BitwiseNot, n, opBitwiseNot), nil
}
func (f *Function) IsFinite(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	isFiniteFn := func(a *bridge.Array, s *bridge.Stream) *bridge.Array {
		ii := bridge.IsInf(a, s)
		in := bridge.IsNaN(a, s)
		ior := bridge.LogicalOr(ii, in, s)
		res := bridge.LogicalNot(ior, s)
		ii.Free()
		in.Free()
		ior.Free()
		return res
	}
	node := f.unaryBoolOp(isFiniteFn, n)
	// Compound: IsInf → t0, IsNaN → t1, LogicalOr → t2, LogicalNot → out
	xi := int32(n.tapeIdx)
	t0 := f.nextTempSlot()
	t1 := f.nextTempSlot()
	t2 := f.nextTempSlot()
	f.emitInstr(opIsInf, t0, []int32{xi})
	f.emitInstr(opIsNaN, t1, []int32{xi})
	f.emitInstr(opLogicalOr, t2, []int32{t0, t1})
	f.emitInstr(opLogicalNot, int32(node.tapeIdx), []int32{t2})
	f.emitInstr(opFreeTemp, -1, []int32{t0, t1, t2})
	return node, nil
}
func (f *Function) IsNaN(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.unaryBoolOp(bridge.IsNaN, n, opIsNaN), nil
}
func (f *Function) Identity(x backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	zero := f.makeScalarConst(0, n.shape.DType)
	r := bridge.Add(n.array, zero, f.stream())
	xi := n.tapeIdx
	dt := n.shape.DType
	node := f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		z := bridge.NewArrayScalarFloat32(0)
		if dt != dtypes.Float32 {
			z2 := bridge.AsType(z, gomlxDTypeToMLX(dt), s)
			z.Free()
			z = z2
		}
		return bridge.Add(arrays[xi], z, s)
	})
	// Compound: new_scalar(0) → t0, optionally astype → t1, add(x, t) → out
	t0 := f.nextTempSlot()
	f.emitInstr(opNewScalarF32, t0, nil, float32Bits(0))
	zeroSlot := t0
	temps := []int32{t0}
	if dt != dtypes.Float32 {
		t1 := f.nextTempSlot()
		f.emitInstr(opAsType, t1, []int32{t0}, int32(gomlxDTypeToMLX(dt)))
		zeroSlot = t1
		temps = append(temps, t1)
	}
	f.emitInstr(opAdd, int32(node.tapeIdx), []int32{int32(xi), zeroSlot})
	f.emitInstr(opFreeTemp, -1, temps)
	return node, nil
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func (f *Function) binaryOp(name string, fn func(*bridge.Array, *bridge.Array, *bridge.Stream) *bridge.Array, lhs, rhs backends.Value, opcodes ...int32) (backends.Value, error) {
	l, err := f.resolveNode(lhs)
	if err != nil {
		return nil, errors.Wrapf(err, "%s lhs", name)
	}
	r, err := f.resolveNode(rhs)
	if err != nil {
		return nil, errors.Wrapf(err, "%s rhs", name)
	}
	result := fn(l.array, r.array, f.stream())
	outShape, _ := shapeinference.BinaryOp(backends.OpTypeAdd, l.shape, r.shape)
	li, ri := l.tapeIdx, r.tapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return fn(arrays[li], arrays[ri], s)
	})
	if len(opcodes) > 0 {
		f.emitInstr(opcodes[0], int32(node.tapeIdx), []int32{int32(li), int32(ri)})
	}
	return node, nil
}

func (f *Function) binaryCompareOp(name string, fn func(*bridge.Array, *bridge.Array, *bridge.Stream) *bridge.Array, lhs, rhs backends.Value, opcodes ...int32) (backends.Value, error) {
	l, _ := f.resolveNode(lhs)
	r, _ := f.resolveNode(rhs)
	result := fn(l.array, r.array, f.stream())
	outShape, _ := shapeinference.ComparisonOp(backends.OpTypeEqual, l.shape, r.shape)
	li, ri := l.tapeIdx, r.tapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return fn(arrays[li], arrays[ri], s)
	})
	if len(opcodes) > 0 {
		f.emitInstr(opcodes[0], int32(node.tapeIdx), []int32{int32(li), int32(ri)})
	}
	return node, nil
}

func (f *Function) Add(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Add", bridge.Add, lhs, rhs, opAdd)
}
func (f *Function) Sub(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Sub", bridge.Subtract, lhs, rhs, opSub)
}
func (f *Function) Mul(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Mul", bridge.Multiply, lhs, rhs, opMul)
}
func (f *Function) Div(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Div", bridge.Divide, lhs, rhs, opDiv)
}
func (f *Function) Rem(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Rem", bridge.Remainder, lhs, rhs, opRem)
}
func (f *Function) Pow(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Pow", bridge.Power, lhs, rhs, opPow)
}
func (f *Function) Max(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Max", bridge.Maximum, lhs, rhs, opMax)
}
func (f *Function) Min(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Min", bridge.Minimum, lhs, rhs, opMin)
}
func (f *Function) Atan2(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Atan2", bridge.Arctan2, lhs, rhs, opAtan2)
}

func (f *Function) LogicalAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("LogicalAnd", bridge.LogicalAnd, lhs, rhs, opLogicalAnd)
}
func (f *Function) LogicalOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("LogicalOr", bridge.LogicalOr, lhs, rhs, opLogicalOr)
}
func (f *Function) LogicalXor(lhs, rhs backends.Value) (backends.Value, error) {
	l, _ := f.resolveNode(lhs)
	r, _ := f.resolveNode(rhs)
	s := f.stream()
	orr := bridge.LogicalOr(l.array, r.array, s)
	andd := bridge.LogicalAnd(l.array, r.array, s)
	notAnd := bridge.LogicalNot(andd, s)
	result := bridge.LogicalAnd(orr, notAnd, s)
	orr.Free()
	andd.Free()
	notAnd.Free()
	outShape, _ := shapeinference.ComparisonOp(backends.OpTypeEqual, l.shape, r.shape)
	li, ri := l.tapeIdx, r.tapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		o := bridge.LogicalOr(arrays[li], arrays[ri], s)
		a := bridge.LogicalAnd(arrays[li], arrays[ri], s)
		na := bridge.LogicalNot(a, s)
		r := bridge.LogicalAnd(o, na, s)
		o.Free()
		a.Free()
		na.Free()
		return r
	})
	// Compound: or → t0, and → t1, not → t2, and(t0,t2) → out
	t0 := f.nextTempSlot()
	t1 := f.nextTempSlot()
	t2 := f.nextTempSlot()
	f.emitInstr(opLogicalOr, t0, []int32{int32(li), int32(ri)})
	f.emitInstr(opLogicalAnd, t1, []int32{int32(li), int32(ri)})
	f.emitInstr(opLogicalNot, t2, []int32{t1})
	f.emitInstr(opLogicalAnd, int32(node.tapeIdx), []int32{t0, t2})
	f.emitInstr(opFreeTemp, -1, []int32{t0, t1, t2})
	return node, nil
}

func (f *Function) BitwiseAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseAnd", bridge.BitwiseAnd, lhs, rhs, opBitwiseAnd)
}
func (f *Function) BitwiseOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseOr", bridge.BitwiseOr, lhs, rhs, opBitwiseOr)
}
func (f *Function) BitwiseXor(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseXor", bridge.BitwiseXor, lhs, rhs, opBitwiseXor)
}
func (f *Function) ShiftLeft(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("ShiftLeft", bridge.LeftShift, lhs, rhs, opLeftShift)
}
func (f *Function) ShiftRightArithmetic(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("ShiftRightArithmetic", bridge.RightShift, lhs, rhs, opRightShift)
}
func (f *Function) ShiftRightLogical(lhs, rhs backends.Value) (backends.Value, error) {
	l, _ := f.resolveNode(lhs)
	r, _ := f.resolveNode(rhs)
	unsignedDType := signedToUnsigned(l.shape.DType)
	if unsignedDType == l.shape.DType {
		return f.binaryOp("ShiftRightLogical", bridge.RightShift, lhs, rhs, opRightShift)
	}
	s := f.stream()
	uMLX := gomlxDTypeToMLX(unsignedDType)
	oMLX := gomlxDTypeToMLX(l.shape.DType)
	lu := bridge.AsType(l.array, uMLX, s)
	shifted := bridge.RightShift(lu, r.array, s)
	result := bridge.AsType(shifted, oMLX, s)
	outShape, _ := shapeinference.BinaryOp(backends.OpTypeShiftRightLogical, l.shape, r.shape)
	li, ri := l.tapeIdx, r.tapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		lu := bridge.AsType(arrays[li], uMLX, s)
		shifted := bridge.RightShift(lu, arrays[ri], s)
		res := bridge.AsType(shifted, oMLX, s)
		lu.Free()
		shifted.Free()
		return res
	})
	// Compound: astype → t0, right_shift → t1, astype → out
	t0 := f.nextTempSlot()
	t1 := f.nextTempSlot()
	f.emitInstr(opAsType, t0, []int32{int32(li)}, int32(uMLX))
	f.emitInstr(opRightShift, t1, []int32{t0, int32(ri)})
	f.emitInstr(opAsType, int32(node.tapeIdx), []int32{t1}, int32(oMLX))
	f.emitInstr(opFreeTemp, -1, []int32{t0, t1})
	return node, nil
}

// signedToUnsigned maps signed integer dtypes to their unsigned counterparts.
func signedToUnsigned(dt dtypes.DType) dtypes.DType {
	switch dt {
	case dtypes.Int8:
		return dtypes.Uint8
	case dtypes.Int16:
		return dtypes.Uint16
	case dtypes.Int32:
		return dtypes.Uint32
	case dtypes.Int64:
		return dtypes.Uint64
	default:
		return dt // Already unsigned or non-integer.
	}
}

// Comparison
func (f *Function) Equal(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("Equal", bridge.Equal, lhs, rhs, opEq)
}
func (f *Function) NotEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("NotEqual", bridge.NotEqual, lhs, rhs, opNe)
}
func (f *Function) LessThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("LessThan", bridge.Less, lhs, rhs, opLt)
}
func (f *Function) LessOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("LessOrEqual", bridge.LessEqual, lhs, rhs, opLe)
}
func (f *Function) GreaterThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("GreaterThan", bridge.Greater, lhs, rhs, opGt)
}
func (f *Function) GreaterOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryCompareOp("GreaterOrEqual", bridge.GreaterEqual, lhs, rhs, opGe)
}

// TotalOrder comparisons — treat same as regular for MLX.
func (f *Function) EqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.Equal(lhs, rhs)
}
func (f *Function) NotEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.NotEqual(lhs, rhs)
}
func (f *Function) LessThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.LessThan(lhs, rhs)
}
func (f *Function) LessOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.LessOrEqual(lhs, rhs)
}
func (f *Function) GreaterThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.GreaterThan(lhs, rhs)
}
func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.GreaterOrEqual(lhs, rhs)
}

// ===========================================================================
// Shape Operations
// ===========================================================================

func (f *Function) Reshape(x backends.Value, dimensions ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	r := bridge.Reshape(n.array, dimensions, f.stream())
	outShape := shapes.Make(n.shape.DType, dimensions...)
	xi := n.tapeIdx
	dims := append([]int{}, dimensions...)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Reshape(arrays[xi], dims, s)
	})
	// params: [ndim, dims...]
	p := []int32{int32(len(dims))}
	for _, d := range dims {
		p = append(p, int32(d))
	}
	f.emitInstr(opReshape, int32(node.tapeIdx), []int32{int32(xi)}, p...)
	return node, nil
}

func (f *Function) Transpose(x backends.Value, permutation ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	r := bridge.Transpose(n.array, permutation, f.stream())
	newDims := make([]int, len(permutation))
	for i, p := range permutation {
		newDims[i] = n.shape.Dimensions[p]
	}
	outShape := shapes.Make(n.shape.DType, newDims...)
	xi := n.tapeIdx
	perm := append([]int{}, permutation...)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Transpose(arrays[xi], perm, s)
	})
	p := []int32{int32(len(perm))}
	for _, a := range perm {
		p = append(p, int32(a))
	}
	f.emitInstr(opTranspose, int32(node.tapeIdx), []int32{int32(xi)}, p...)
	return node, nil
}

func (f *Function) BroadcastInDim(x backends.Value, outputShape shapes.Shape, broadcastAxes []int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	expandedDims := make([]int, outputShape.Rank())
	for i := range expandedDims {
		expandedDims[i] = 1
	}
	for i, axis := range broadcastAxes {
		expandedDims[axis] = n.shape.Dimensions[i]
	}
	s := f.stream()
	reshaped := bridge.Reshape(n.array, expandedDims, s)
	r := bridge.BroadcastTo(reshaped, outputShape.Dimensions, s)
	xi := n.tapeIdx
	eDims := append([]int{}, expandedDims...)
	oDims := append([]int{}, outputShape.Dimensions...)
	node := f.record(outputShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		rs := bridge.Reshape(arrays[xi], eDims, s)
		return bridge.BroadcastTo(rs, oDims, s)
	})
	// Compound: reshape → t0, broadcast_to → out
	t0 := f.nextTempSlot()
	rp := []int32{int32(len(eDims))}
	for _, d := range eDims {
		rp = append(rp, int32(d))
	}
	f.emitInstr(opReshape, t0, []int32{int32(xi)}, rp...)
	bp := []int32{int32(len(oDims))}
	for _, d := range oDims {
		bp = append(bp, int32(d))
	}
	f.emitInstr(opBroadcastTo, int32(node.tapeIdx), []int32{t0}, bp...)
	f.emitInstr(opFreeTemp, -1, []int32{t0})
	return node, nil
}

func (f *Function) ConvertDType(x backends.Value, dtype dtypes.DType) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	mlxDt := gomlxDTypeToMLX(dtype)
	r := bridge.AsType(n.array, mlxDt, f.stream())
	outShape := n.shape.Clone()
	outShape.DType = dtype
	xi := n.tapeIdx
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.AsType(arrays[xi], mlxDt, s)
	})
	f.emitInstr(opAsType, int32(node.tapeIdx), []int32{int32(xi)}, int32(mlxDt))
	return node, nil
}

func (f *Function) Where(condition, onTrue, onFalse backends.Value) (backends.Value, error) {
	c, _ := f.resolveNode(condition)
	t, _ := f.resolveNode(onTrue)
	fa, _ := f.resolveNode(onFalse)
	r := bridge.Where(c.array, t.array, fa.array, f.stream())
	outShape, _ := shapeinference.WhereOp(c.shape, t.shape, fa.shape)
	ci, ti, fi := c.tapeIdx, t.tapeIdx, fa.tapeIdx
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Where(arrays[ci], arrays[ti], arrays[fi], s)
	})
	f.emitInstr(opWhere, int32(node.tapeIdx), []int32{int32(ci), int32(ti), int32(fi)})
	return node, nil
}

func (f *Function) Clamp(min, x, max backends.Value) (backends.Value, error) {
	mn, _ := f.resolveNode(min)
	n, _ := f.resolveNode(x)
	mx, _ := f.resolveNode(max)
	r := bridge.Clip(n.array, mn.array, mx.array, f.stream())
	mni, ni, mxi := mn.tapeIdx, n.tapeIdx, mx.tapeIdx
	node := f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Clip(arrays[ni], arrays[mni], arrays[mxi], s)
	})
	f.emitInstr(opClip, int32(node.tapeIdx), []int32{int32(ni), int32(mni), int32(mxi)})
	return node, nil
}

func (f *Function) Slice(x backends.Value, starts, limits, strides []int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	r := bridge.Slice(n.array, starts, limits, strides, f.stream())
	outDims := make([]int, len(starts))
	for i := range starts {
		outDims[i] = (limits[i] - starts[i] + strides[i] - 1) / strides[i]
	}
	outShape := shapes.Make(n.shape.DType, outDims...)
	xi := n.tapeIdx
	st := append([]int{}, starts...)
	li := append([]int{}, limits...)
	sr := append([]int{}, strides...)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Slice(arrays[xi], st, li, sr, s)
	})
	// params: [ndim, starts..., stops..., strides...]
	p := []int32{int32(len(st))}
	for _, v := range st {
		p = append(p, int32(v))
	}
	for _, v := range li {
		p = append(p, int32(v))
	}
	for _, v := range sr {
		p = append(p, int32(v))
	}
	f.emitInstr(opSlice, int32(node.tapeIdx), []int32{int32(xi)}, p...)
	return node, nil
}

func (f *Function) Concatenate(axis int, operands ...backends.Value) (backends.Value, error) {
	nodes, err := f.resolveNodes("Concatenate", operands...)
	if err != nil {
		return nil, err
	}
	arrays := make([]*bridge.Array, len(nodes))
	tapeIndices := make([]int, len(nodes))
	for i, n := range nodes {
		arrays[i] = n.array
		tapeIndices[i] = n.tapeIdx
	}
	r := bridge.Concatenate(arrays, axis, f.stream())
	totalAxisDim := 0
	for _, n := range nodes {
		totalAxisDim += n.shape.Dimensions[axis]
	}
	outDims := make([]int, len(nodes[0].shape.Dimensions))
	copy(outDims, nodes[0].shape.Dimensions)
	outDims[axis] = totalAxisDim
	outShape := shapes.Make(nodes[0].shape.DType, outDims...)
	ax := axis
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		arrs := make([]*bridge.Array, len(tapeIndices))
		for i, ti := range tapeIndices {
			arrs[i] = arrays[ti]
		}
		return bridge.Concatenate(arrs, ax, s)
	})
	ins := make([]int32, len(tapeIndices))
	for i, ti := range tapeIndices {
		ins[i] = int32(ti)
	}
	f.emitInstr(opConcat, int32(node.tapeIdx), ins, int32(ax))
	return node, nil
}

func (f *Function) Reverse(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	r := bridge.Flip(n.array, axes, f.stream())
	xi := n.tapeIdx
	ax := append([]int{}, axes...)
	node := f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Flip(arrays[xi], ax, s)
	})
	p := []int32{int32(len(ax))}
	for _, a := range ax {
		p = append(p, int32(a))
	}
	f.emitInstr(opFlip, int32(node.tapeIdx), []int32{int32(xi)}, p...)
	return node, nil
}

func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (backends.Value, error) {
	s := f.stream()
	axisSize := shape.Dimensions[iotaAxis]
	arr := bridge.Arange(0, float64(axisSize), 1, gomlxDTypeToMLX(shape.DType), s)
	reshapeDims := make([]int, shape.Rank())
	for i := range reshapeDims {
		reshapeDims[i] = 1
	}
	reshapeDims[iotaAxis] = axisSize
	reshaped := bridge.Reshape(arr, reshapeDims, s)
	arr.Free()
	r := bridge.BroadcastTo(reshaped, shape.Dimensions, s)
	reshaped.Free()
	// Iota is a constant — return self during replay.
	iotaArr := r
	node := f.record(shape, r, func(_ []*bridge.Array, _ *bridge.Stream) *bridge.Array {
		return iotaArr
	})
	f.constIndices = append(f.constIndices, node.tapeIdx)
	return node, nil
}

func (f *Function) Pad(x, fillValue backends.Value, axesConfig ...backends.PadAxis) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	fill, _ := f.resolveNode(fillValue)

	axes := make([]int, len(axesConfig))
	lowPads := make([]int, len(axesConfig))
	highPads := make([]int, len(axesConfig))
	hasInterior := false
	for i, ac := range axesConfig {
		axes[i] = i
		lowPads[i] = ac.Start
		highPads[i] = ac.End
		if ac.Interior > 0 {
			hasInterior = true
		}
	}
	if hasInterior {
		return nil, errors.Errorf("Pad: interior padding not yet supported")
	}

	r := bridge.Pad(n.array, fill.array, axes, lowPads, highPads, f.stream())
	outDims := make([]int, len(n.shape.Dimensions))
	for i, d := range n.shape.Dimensions {
		if i < len(axesConfig) {
			outDims[i] = d + axesConfig[i].Start + axesConfig[i].End
		} else {
			outDims[i] = d
		}
	}
	outShape := shapes.Make(n.shape.DType, outDims...)
	ni, fi := n.tapeIdx, fill.tapeIdx
	ax := append([]int{}, axes...)
	lp := append([]int{}, lowPads...)
	hp := append([]int{}, highPads...)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Pad(arrays[ni], arrays[fi], ax, lp, hp, s)
	})
	// params: [naxes, axes..., low_pads..., high_pads...]
	p := []int32{int32(len(ax))}
	for _, a := range ax {
		p = append(p, int32(a))
	}
	for _, v := range lp {
		p = append(p, int32(v))
	}
	for _, v := range hp {
		p = append(p, int32(v))
	}
	f.emitInstr(opPad, int32(node.tapeIdx), []int32{int32(ni), int32(fi)}, p...)
	return node, nil
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

// reduceOp is a helper for reduction operations with tape recording.
func (f *Function) reduceOp(fn func(*bridge.Array, []int, bool, *bridge.Stream) *bridge.Array, x *graphNode, axes []int, opcodes ...int32) *graphNode {
	// Empty axes means "reduce all dimensions" in GoMLX convention.
	// MLX interprets empty axes as no-op, so expand to all axes explicitly.
	if len(axes) == 0 {
		axes = make([]int, x.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
	}
	r := fn(x.array, axes, false, f.stream())
	outShape, _ := shapeinference.ReduceOp(x.shape, axes)
	xi := x.tapeIdx
	ax := append([]int{}, axes...)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return fn(arrays[xi], ax, false, s)
	})
	if len(opcodes) > 0 {
		// params: [naxes, axes..., keepdims]
		p := []int32{int32(len(ax))}
		for _, a := range ax {
			p = append(p, int32(a))
		}
		p = append(p, 0) // keepdims=false
		f.emitInstr(opcodes[0], int32(node.tapeIdx), []int32{int32(xi)}, p...)
	}
	return node
}

func (f *Function) ReduceSum(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.reduceOp(bridge.Sum, n, axes, opReduceSum), nil
}
func (f *Function) ReduceMax(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.reduceOp(bridge.Max, n, axes, opReduceMax), nil
}
func (f *Function) ReduceMin(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.reduceOp(bridge.Min, n, axes, opReduceMin), nil
}
func (f *Function) ReduceProduct(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	return f.reduceOp(bridge.Prod, n, axes, opReduceProd), nil
}
func (f *Function) ReduceLogicalAnd(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	node := f.reduceOp(bridge.All, n, axes, opReduceAll)
	node.shape.DType = dtypes.Bool
	return node, nil
}
func (f *Function) ReduceLogicalOr(x backends.Value, axes ...int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	node := f.reduceOp(bridge.Any, n, axes, opReduceAny)
	node.shape.DType = dtypes.Bool
	return node, nil
}

func (f *Function) ArgMinMax(x backends.Value, axis int, outputDType dtypes.DType, isMin bool) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	var r *bridge.Array
	if isMin {
		r = bridge.ArgMin(n.array, axis, false, f.stream())
	} else {
		r = bridge.ArgMax(n.array, axis, false, f.stream())
	}
	if mlxDTypeToGoMLX(r.DType()) != outputDType {
		casted := bridge.AsType(r, gomlxDTypeToMLX(outputDType), f.stream())
		r.Free()
		r = casted
	}
	outDims := make([]int, 0, len(n.shape.Dimensions)-1)
	for i, d := range n.shape.Dimensions {
		if i != axis {
			outDims = append(outDims, d)
		}
	}
	outShape := shapes.Make(outputDType, outDims...)
	xi := n.tapeIdx
	ax := axis
	mlxDt := gomlxDTypeToMLX(outputDType)
	node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		var res *bridge.Array
		if isMin {
			res = bridge.ArgMin(arrays[xi], ax, false, s)
		} else {
			res = bridge.ArgMax(arrays[xi], ax, false, s)
		}
		if mlxDTypeToGoMLX(res.DType()) != outputDType {
			casted := bridge.AsType(res, mlxDt, s)
			res.Free()
			return casted
		}
		return res
	})
	// params: [axis, keepdims, out_dtype_or_-1]
	oc := opArgMax
	if isMin {
		oc = opArgMin
	}
	dtParam := int32(-1) // no cast needed
	if mlxDTypeToGoMLX(r.DType()) != outputDType {
		dtParam = int32(mlxDt)
	}
	f.emitInstr(oc, int32(node.tapeIdx), []int32{int32(xi)}, int32(ax), 0, dtParam)
	return node, nil
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

func (f *Function) Dot(lhs, rhs backends.Value) (backends.Value, error) {
	l, _ := f.resolveNode(lhs)
	r, _ := f.resolveNode(rhs)
	result := bridge.MatMul(l.array, r.array, f.stream())
	lDims := l.shape.Dimensions
	rDims := r.shape.Dimensions
	var outDims []int
	if len(lDims) == 1 && len(rDims) == 1 {
		outDims = nil
	} else if len(lDims) == 2 && len(rDims) == 2 {
		outDims = []int{lDims[0], rDims[1]}
	} else {
		outDims = []int{lDims[0], rDims[len(rDims)-1]}
	}
	outShape := shapes.Make(l.shape.DType, outDims...)
	li, ri := l.tapeIdx, r.tapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.MatMul(arrays[li], arrays[ri], s)
	})
	f.emitInstr(opMatMul, int32(node.tapeIdx), []int32{int32(li), int32(ri)})
	return node, nil
}

func (f *Function) DotGeneral(lhs backends.Value, lhsContractingAxes, lhsBatchAxes []int,
	rhs backends.Value, rhsContractingAxes, rhsBatchAxes []int,
	config backends.DotGeneralConfig) (backends.Value, error) {

	l, _ := f.resolveNode(lhs)
	r, _ := f.resolveNode(rhs)

	outShape := dotGeneralShape(l.shape, lhsContractingAxes, lhsBatchAxes, r.shape, rhsContractingAxes, rhsBatchAxes)
	li, ri := l.tapeIdx, r.tapeIdx

	if isSimpleMatMul(l.shape, r.shape, lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes) {
		result := bridge.MatMul(l.array, r.array, f.stream())
		node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			return bridge.MatMul(arrays[li], arrays[ri], s)
		})
		f.emitInstr(opMatMul, int32(node.tapeIdx), []int32{int32(li), int32(ri)})
		return node, nil
	}

	// General case: decompose via transpose + reshape + matmul + reshape.
	// Emit as C tape instructions to avoid Go callback path (which leaked
	// intermediate arrays — bridge.Array has no GC finalizer).
	lCross := crossAxes(l.shape.Rank(), lhsContractingAxes, lhsBatchAxes)
	rCross := crossAxes(r.shape.Rank(), rhsContractingAxes, rhsBatchAxes)

	// Compute permutations: [batch..., cross..., contract...] for LHS,
	// [batch..., contract..., cross...] for RHS.
	lPerm := append(append(append([]int{}, lhsBatchAxes...), lCross...), lhsContractingAxes...)
	rPerm := append(append(append([]int{}, rhsBatchAxes...), rhsContractingAxes...), rCross...)

	// Compute dimension sizes.
	batchSize := 1
	for _, ax := range lhsBatchAxes {
		batchSize *= l.shape.Dimensions[ax]
	}
	lCrossSize := 1
	for _, ax := range lCross {
		lCrossSize *= l.shape.Dimensions[ax]
	}
	contractSize := 1
	for _, ax := range lhsContractingAxes {
		contractSize *= l.shape.Dimensions[ax]
	}
	rCrossSize := 1
	for _, ax := range rCross {
		rCrossSize *= r.shape.Dimensions[ax]
	}

	// Build-time: compute the result using bridge ops.
	s := f.stream()
	lT := bridge.Transpose(l.array, lPerm, s)
	rT := bridge.Transpose(r.array, rPerm, s)
	var lR, rR *bridge.Array
	if batchSize > 1 {
		lR = bridge.Reshape(lT, []int{batchSize, lCrossSize, contractSize}, s)
		rR = bridge.Reshape(rT, []int{batchSize, contractSize, rCrossSize}, s)
	} else {
		lR = bridge.Reshape(lT, []int{lCrossSize, contractSize}, s)
		rR = bridge.Reshape(rT, []int{contractSize, rCrossSize}, s)
	}
	mm := bridge.MatMul(lR, rR, s)
	result := bridge.Reshape(mm, outShape.Dimensions, s)
	f.buildArrays = append(f.buildArrays, lT, rT, lR, rR, mm, result)

	// Emit C tape instructions: transpose → reshape → matmul → reshape.
	// Use temp slots for intermediates.
	tLT := f.nextTempSlot()
	tRT := f.nextTempSlot()
	tLR := f.nextTempSlot()
	tRR := f.nextTempSlot()
	tMM := f.nextTempSlot()

	// LHS transpose — params format: [ndim, perm...]
	lPermParams := []int32{int32(len(lPerm))}
	for _, p := range lPerm {
		lPermParams = append(lPermParams, int32(p))
	}
	f.emitInstr(opTranspose, tLT, []int32{int32(li)}, lPermParams...)

	// RHS transpose — params format: [ndim, perm...]
	rPermParams := []int32{int32(len(rPerm))}
	for _, p := range rPerm {
		rPermParams = append(rPermParams, int32(p))
	}
	f.emitInstr(opTranspose, tRT, []int32{int32(ri)}, rPermParams...)

	// LHS reshape — params format: [ndim, dims...]
	if batchSize > 1 {
		f.emitInstr(opReshape, tLR, []int32{tLT}, int32(3), int32(batchSize), int32(lCrossSize), int32(contractSize))
	} else {
		f.emitInstr(opReshape, tLR, []int32{tLT}, int32(2), int32(lCrossSize), int32(contractSize))
	}

	// RHS reshape — params format: [ndim, dims...]
	if batchSize > 1 {
		f.emitInstr(opReshape, tRR, []int32{tRT}, int32(3), int32(batchSize), int32(contractSize), int32(rCrossSize))
	} else {
		f.emitInstr(opReshape, tRR, []int32{tRT}, int32(2), int32(contractSize), int32(rCrossSize))
	}

	// Free transpose temps (no longer needed after reshape)
	f.emitInstr(opFreeTemp, -1, []int32{tLT, tRT})

	// MatMul
	f.emitInstr(opMatMul, tMM, []int32{tLR, tRR})

	// Free reshape temps
	f.emitInstr(opFreeTemp, -1, []int32{tLR, tRR})

	// Final reshape to output shape — params format: [ndim, dims...]
	outReshapeParams := []int32{int32(len(outShape.Dimensions))}
	for _, d := range outShape.Dimensions {
		outReshapeParams = append(outReshapeParams, int32(d))
	}
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return dotGeneralReplay(arrays[li], l.shape, lhsContractingAxes, lhsBatchAxes,
			arrays[ri], r.shape, rhsContractingAxes, rhsBatchAxes, outShape, s)
	})
	f.emitInstr(opReshape, int32(node.tapeIdx), []int32{tMM}, outReshapeParams...)

	// Free matmul temp
	f.emitInstr(opFreeTemp, -1, []int32{tMM})

	return node, nil
}

// dotGeneralShape computes the output shape for a DotGeneral operation.
func dotGeneralShape(lShape shapes.Shape, lContract, lBatch []int, rShape shapes.Shape, rContract, rBatch []int) shapes.Shape {
	isContracting := func(ax int, contract []int) bool {
		for _, c := range contract {
			if c == ax {
				return true
			}
		}
		return false
	}
	isBatch := func(ax int, batch []int) bool {
		for _, b := range batch {
			if b == ax {
				return true
			}
		}
		return false
	}

	var dims []int
	// Batch dimensions (from LHS).
	for _, ax := range lBatch {
		dims = append(dims, lShape.Dimensions[ax])
	}
	// LHS cross dimensions (not contracting, not batch).
	for ax := 0; ax < lShape.Rank(); ax++ {
		if !isContracting(ax, lContract) && !isBatch(ax, lBatch) {
			dims = append(dims, lShape.Dimensions[ax])
		}
	}
	// RHS cross dimensions.
	for ax := 0; ax < rShape.Rank(); ax++ {
		if !isContracting(ax, rContract) && !isBatch(ax, rBatch) {
			dims = append(dims, rShape.Dimensions[ax])
		}
	}
	return shapes.Make(lShape.DType, dims...)
}

// isSimpleMatMul checks if the DotGeneral is a standard matmul.
func isSimpleMatMul(lShape, rShape shapes.Shape, lContract, lBatch, rContract, rBatch []int) bool {
	if len(lBatch) > 0 || len(rBatch) > 0 {
		return false
	}
	if lShape.Rank() != 2 || rShape.Rank() != 2 {
		return false
	}
	if len(lContract) != 1 || len(rContract) != 1 {
		return false
	}
	return lContract[0] == 1 && rContract[0] == 0
}

// dotGeneralDecompose handles the general case by delegating to dotGeneralReplay
// with the node's arrays and shapes.
func (f *Function) dotGeneralDecompose(
	l *graphNode, lContract, lBatch []int,
	r *graphNode, rContract, rBatch []int,
	outShape shapes.Shape,
) (*graphNode, error) {
	result := dotGeneralReplay(l.array, l.shape, lContract, lBatch, r.array, r.shape, rContract, rBatch, outShape, f.stream())
	f.buildArrays = append(f.buildArrays, result)
	return &graphNode{array: result, shape: outShape, owner: f}, nil
}

// crossAxes returns axes that are neither contracting nor batch.
func crossAxes(rank int, contract, batch []int) []int {
	used := make(map[int]bool)
	for _, a := range contract {
		used[a] = true
	}
	for _, a := range batch {
		used[a] = true
	}
	var cross []int
	for i := 0; i < rank; i++ {
		if !used[i] {
			cross = append(cross, i)
		}
	}
	return cross
}

// dotGeneralReplay replays a DotGeneral decomposition during tape execution.
func dotGeneralReplay(lArr *bridge.Array, lShape shapes.Shape, lContract, lBatch []int,
	rArr *bridge.Array, rShape shapes.Shape, rContract, rBatch []int,
	outShape shapes.Shape, s *bridge.Stream) *bridge.Array {

	lCross := crossAxes(lShape.Rank(), lContract, lBatch)
	rCross := crossAxes(rShape.Rank(), rContract, rBatch)

	lPerm := append(append(append([]int{}, lBatch...), lCross...), lContract...)
	lT := bridge.Transpose(lArr, lPerm, s)
	rPerm := append(append(append([]int{}, rBatch...), rContract...), rCross...)
	rT := bridge.Transpose(rArr, rPerm, s)

	batchSize := 1
	for _, ax := range lBatch {
		batchSize *= lShape.Dimensions[ax]
	}
	lCrossSize := 1
	for _, ax := range lCross {
		lCrossSize *= lShape.Dimensions[ax]
	}
	contractSize := 1
	for _, ax := range lContract {
		contractSize *= lShape.Dimensions[ax]
	}
	rCrossSize := 1
	for _, ax := range rCross {
		rCrossSize *= rShape.Dimensions[ax]
	}

	if batchSize > 1 {
		lTr := bridge.Reshape(lT, []int{batchSize, lCrossSize, contractSize}, s)
		lT.Free()
		lT = lTr
		rTr := bridge.Reshape(rT, []int{batchSize, contractSize, rCrossSize}, s)
		rT.Free()
		rT = rTr
	} else {
		lTr := bridge.Reshape(lT, []int{lCrossSize, contractSize}, s)
		lT.Free()
		lT = lTr
		rTr := bridge.Reshape(rT, []int{contractSize, rCrossSize}, s)
		rT.Free()
		rT = rTr
	}

	result := bridge.MatMul(lT, rT, s)
	lT.Free()
	rT.Free()
	final := bridge.Reshape(result, outShape.Dimensions, s)
	result.Free()
	return final
}

// ===========================================================================
// Dynamic Slice Operations
// ===========================================================================

// readScalarInt reads a scalar integer value from an MLX array, handling all integer dtypes.
func readScalarInt(arr *bridge.Array, dt dtypes.DType) int {
	ptr := arr.DataPtr()
	switch dt {
	case dtypes.Int32:
		return int(*(*int32)(ptr))
	case dtypes.Int64:
		return int(*(*int64)(ptr))
	case dtypes.Int16:
		return int(*(*int16)(ptr))
	case dtypes.Int8:
		return int(*(*int8)(ptr))
	case dtypes.Uint32:
		return int(*(*uint32)(ptr))
	case dtypes.Uint64:
		return int(*(*uint64)(ptr))
	case dtypes.Uint16:
		return int(*(*uint16)(ptr))
	case dtypes.Uint8:
		return int(*(*uint8)(ptr))
	default:
		// Fallback to int32 for unexpected types.
		return int(*(*int32)(ptr))
	}
}

// newScalarSeed creates a scalar MLX array for a seed value in the given dtype,
// avoiding int32 truncation for larger integer types.
func newScalarSeed(val uint64, dt dtypes.DType) *bridge.Array {
	switch dt {
	case dtypes.Int64:
		v := int64(val)
		return bridge.NewArrayFromData(unsafe.Pointer(&v), nil, gomlxDTypeToMLX(dt))
	case dtypes.Uint64:
		v := val
		return bridge.NewArrayFromData(unsafe.Pointer(&v), nil, gomlxDTypeToMLX(dt))
	case dtypes.Uint32:
		v := uint32(val)
		return bridge.NewArrayFromData(unsafe.Pointer(&v), nil, gomlxDTypeToMLX(dt))
	default:
		return bridge.NewArrayScalarInt32(int32(val))
	}
}

// dynamicIndexInfo holds the tape metadata for dynamic index nodes,
// used to reconstruct start positions during tape replay.
type dynamicIndexInfo struct {
	tapeIdx int
	dtype   dtypes.DType
}

// evalDynamicIndices resolves, evaluates, and reads the current integer value of
// each start-index node. It returns the per-axis metadata needed for tape replay
// and the concrete start positions for the current execution.
func (f *Function) evalDynamicIndices(opName string, startIndices []backends.Value) ([]dynamicIndexInfo, []int, error) {
	info := make([]dynamicIndexInfo, len(startIndices))
	starts := make([]int, len(startIndices))
	for i, idx := range startIndices {
		idxNode, _ := f.resolveNode(idx)
		info[i] = dynamicIndexInfo{tapeIdx: idxNode.tapeIdx, dtype: idxNode.shape.DType}
		if err := bridge.Eval(idxNode.array); err != nil {
			return nil, nil, errors.Wrapf(err, "%s: eval start index %d", opName, i)
		}
		starts[i] = readScalarInt(idxNode.array, idxNode.shape.DType)
	}
	return info, starts, nil
}

// replayDynamicIndices reads the concrete start positions from already-evaluated
// tape arrays during tape replay, using the metadata from evalDynamicIndices.
func replayDynamicIndices(arrays []*bridge.Array, info []dynamicIndexInfo) []int {
	starts := make([]int, len(info))
	for i, di := range info {
		bridge.Eval(arrays[di.tapeIdx])
		starts[i] = readScalarInt(arrays[di.tapeIdx], di.dtype)
	}
	return starts
}

func (f *Function) DynamicSlice(operand backends.Value, startIndices []backends.Value, sliceDims []int) (backends.Value, error) {
	f.markGoCallback("DynamicSlice")
	n, _ := f.resolveNode(operand)
	s := f.stream()

	idxInfo, starts, err := f.evalDynamicIndices("DynamicSlice", startIndices)
	if err != nil {
		return nil, err
	}

	stops := make([]int, len(starts))
	strides := make([]int, len(starts))
	for i := range starts {
		stops[i] = starts[i] + sliceDims[i]
		strides[i] = 1
	}
	r := bridge.Slice(n.array, starts, stops, strides, s)
	outShape := shapes.Make(n.shape.DType, sliceDims...)
	ni := n.tapeIdx
	sd := append([]int{}, sliceDims...)
	return f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		st := replayDynamicIndices(arrays, idxInfo)
		sp := make([]int, len(st))
		sr := make([]int, len(st))
		for i := range st {
			sp[i] = st[i] + sd[i]
			sr[i] = 1
		}
		return bridge.Slice(arrays[ni], st, sp, sr, s)
	}), nil
}

func (f *Function) DynamicUpdateSlice(operand, update backends.Value, startIndices []backends.Value) (backends.Value, error) {
	f.markGoCallback("DynamicUpdateSlice")
	n, _ := f.resolveNode(operand)
	u, _ := f.resolveNode(update)
	s := f.stream()

	idxInfo, starts, err := f.evalDynamicIndices("DynamicUpdateSlice", startIndices)
	if err != nil {
		return nil, err
	}

	stops := make([]int, len(starts))
	strides := make([]int, len(starts))
	for i := range starts {
		stops[i] = starts[i] + u.shape.Dimensions[i]
		strides[i] = 1
	}
	r := bridge.SliceUpdate(n.array, u.array, starts, stops, strides, s)
	ni, ui := n.tapeIdx, u.tapeIdx
	uDims := append([]int{}, u.shape.Dimensions...)
	return f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		st := replayDynamicIndices(arrays, idxInfo)
		sp := make([]int, len(st))
		sr := make([]int, len(st))
		for i := range st {
			sp[i] = st[i] + uDims[i]
			sr[i] = 1
		}
		return bridge.SliceUpdate(arrays[ni], arrays[ui], st, sp, sr, s)
	}), nil
}

// ===========================================================================
// Gather / Scatter
// ===========================================================================

func (f *Function) Gather(operand, startIndices backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Value, error) {
	op, _ := f.resolveNode(operand)
	idx, _ := f.resolveNode(startIndices)
	s := f.stream()

	// Simple single-axis embedding lookup: use bridge.Take.
	if len(startIndexMap) == 1 && len(collapsedSliceAxes) == 1 &&
		collapsedSliceAxes[0] == startIndexMap[0] {
		axis := startIndexMap[0]
		// Check if all other slice sizes match operand dims.
		isSimple := true
		for i, sz := range sliceSizes {
			if i == axis {
				if sz != 1 {
					isSimple = false
				}
			} else if sz != op.shape.Dimensions[i] {
				isSimple = false
			}
		}
		if isSimple {
			flatIdx := bridge.Reshape(idx.array, []int{idx.array.Size()}, s)
			r := bridge.Take(op.array, flatIdx, axis, s)
			flatIdx.Free()

			outShape, _ := shapeinference.Gather(op.shape, idx.shape, indexVectorAxis,
				offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
			// Take with flat indices produces [N, ...] — reshape to match expected output shape.
			outDims := outShape.Dimensions
			r = bridge.Reshape(r, outDims, s)
			oi, ii := op.tapeIdx, idx.tapeIdx
			idxSize := idx.array.Size()
			node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
				fi := bridge.Reshape(arrays[ii], []int{idxSize}, s)
				res := bridge.Take(arrays[oi], fi, axis, s)
				fi.Free()
				res = bridge.Reshape(res, outDims, s)
				return res
			})
			// Compound: reshape indices → t0, take → t1, reshape output → out
			t0 := f.nextTempSlot()
			t1 := f.nextTempSlot()
			f.emitInstr(opReshape, t0, []int32{int32(ii)}, int32(1), int32(idxSize))
			f.emitInstr(opTake, t1, []int32{int32(oi), t0}, int32(axis))
			// Reshape take result from [flatN, ...] to expected output dims
			reshapeParams := []int32{int32(len(outDims))}
			for _, d := range outDims {
				reshapeParams = append(reshapeParams, int32(d))
			}
			f.emitInstr(opReshape, int32(node.tapeIdx), []int32{t1}, reshapeParams...)
			f.emitInstr(opFreeTemp, -1, []int32{t0})
			f.emitInstr(opFreeTemp, -1, []int32{t1})
			return node, nil
		}
	}

	// General case: fall back to not-implemented for now.
	return nil, errors.Wrapf(backends.ErrNotImplemented, "Gather: complex multi-axis gather not yet supported in MLX backend")
}

func (f *Function) scatterOp(name string, scatterFn func(*bridge.Array, []*bridge.Array, *bridge.Array, []int, *bridge.Stream) *bridge.Array,
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
	opcode int32,
) (backends.Value, error) {
	op, _ := f.resolveNode(operand)
	idx, _ := f.resolveNode(scatterIndices)
	upd, _ := f.resolveNode(updates)
	s := f.stream()

	// Simple single-axis scatter.
	if len(scatterAxesToOperandAxes) == 1 {
		axis := scatterAxesToOperandAxes[0]
		indices := []*bridge.Array{idx.array}
		axes := []int{axis}
		r := scatterFn(op.array, indices, upd.array, axes, s)
		oi, ii, ui := op.tapeIdx, idx.tapeIdx, upd.tapeIdx
		node := f.record(op.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			return scatterFn(arrays[oi], []*bridge.Array{arrays[ii]}, arrays[ui], []int{axis}, s)
		})
		f.emitInstr(opcode, int32(node.tapeIdx), []int32{int32(oi), int32(ii), int32(ui)}, int32(1), int32(axis))
		return node, nil
	}

	f.markGoCallback("Scatter(complex)")
	return nil, errors.Wrapf(backends.ErrNotImplemented, "%s: complex scatter not yet supported", name)
}

func (f *Function) ScatterSum(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterOp("ScatterSum", bridge.ScatterAdd, operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, opScatterAdd)
}

func (f *Function) ScatterMax(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterOp("ScatterMax", bridge.ScatterMax, operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, opScatterMax)
}

func (f *Function) ScatterMin(operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterOp("ScatterMin", bridge.ScatterMin, operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, opScatterMin)
}

// ===========================================================================
// Convolution
// ===========================================================================

func (f *Function) ConvGeneral(
	input, kernel backends.Value,
	axes backends.ConvolveAxesConfig,
	strides []int,
	paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) (backends.Value, error) {
	inp, _ := f.resolveNode(input)
	ker, _ := f.resolveNode(kernel)
	s := f.stream()

	if batchGroupCount > 1 {
		return nil, errors.Errorf("ConvGeneral: batchGroupCount > 1 not supported")
	}
	for _, d := range inputDilations {
		if d > 1 {
			return nil, errors.Errorf("ConvGeneral: input dilation > 1 not supported")
		}
	}

	numSpatial := len(axes.InputSpatial)

	// MLX expects NHWC layout for conv2d. Transpose if needed.
	ii, ki := inp.tapeIdx, ker.tapeIdx
	if numSpatial == 2 {
		stride := [2]int{strides[0], strides[1]}
		padding := [2]int{paddings[0][0], paddings[1][0]}
		dilation := [2]int{kernelDilations[0], kernelDilations[1]}
		r := bridge.Conv2d(inp.array, ker.array, stride, padding, dilation, channelGroupCount, s)
		outShape, err := shapeinference.ConvGeneralOp(inp.shape, ker.shape, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
		if err != nil {
			return nil, err
		}
		groups := channelGroupCount
		node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			return bridge.Conv2d(arrays[ii], arrays[ki], stride, padding, dilation, groups, s)
		})
		f.emitInstr(opConv2d, int32(node.tapeIdx), []int32{int32(ii), int32(ki)},
			int32(stride[0]), int32(stride[1]), int32(padding[0]), int32(padding[1]),
			int32(dilation[0]), int32(dilation[1]), int32(groups))
		return node, nil
	} else if numSpatial == 1 {
		st, pd, dl, groups := strides[0], paddings[0][0], kernelDilations[0], channelGroupCount
		r := bridge.Conv1d(inp.array, ker.array, st, pd, dl, groups, s)
		outShape, err := shapeinference.ConvGeneralOp(inp.shape, ker.shape, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
		if err != nil {
			return nil, err
		}
		node := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			return bridge.Conv1d(arrays[ii], arrays[ki], st, pd, dl, groups, s)
		})
		f.emitInstr(opConv1d, int32(node.tapeIdx), []int32{int32(ii), int32(ki)},
			int32(st), int32(pd), int32(dl), int32(groups))
		return node, nil
	}

	return nil, errors.Errorf("ConvGeneral: only 1D and 2D convolution supported, got %d spatial dims", numSpatial)
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Value,
	epsilon float32, featureAxis int) (backends.Value, error) {
	f.markGoCallback("BatchNormForInference") // compound op — uses Go callback path
	x, _ := f.resolveNode(operand)
	sc, _ := f.resolveNode(scale)
	off, _ := f.resolveNode(offset)
	m, _ := f.resolveNode(mean)
	v, _ := f.resolveNode(variance)
	s := f.stream()

	eps := bridge.NewArrayScalarFloat32(epsilon)
	varPlusEps := bridge.Add(v.array, eps, s)
	invStd := bridge.Rsqrt(varPlusEps, s)
	xCentered := bridge.Subtract(x.array, m.array, s)
	normalized := bridge.Multiply(xCentered, invStd, s)
	scaled := bridge.Multiply(normalized, sc.array, s)
	result := bridge.Add(scaled, off.array, s)

	eps.Free()
	varPlusEps.Free()
	invStd.Free()
	xCentered.Free()
	normalized.Free()
	scaled.Free()

	xi, si, oi, mi, vi := x.tapeIdx, sc.tapeIdx, off.tapeIdx, m.tapeIdx, v.tapeIdx
	ep := epsilon
	return f.record(x.shape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		e := bridge.NewArrayScalarFloat32(ep)
		vpe := bridge.Add(arrays[vi], e, s)
		is := bridge.Rsqrt(vpe, s)
		xc := bridge.Subtract(arrays[xi], arrays[mi], s)
		nm := bridge.Multiply(xc, is, s)
		sc := bridge.Multiply(nm, arrays[si], s)
		res := bridge.Add(sc, arrays[oi], s)
		e.Free()
		vpe.Free()
		is.Free()
		xc.Free()
		nm.Free()
		sc.Free()
		return res
	}), nil
}

func (f *Function) BatchNormForTraining(operand, scale, offset backends.Value,
	epsilon float32, featureAxis int) (normalized, batchMean, batchVariance backends.Value, err error) {
	f.markGoCallback("BatchNormForTraining") // compound op
	x, _ := f.resolveNode(operand)
	sc, _ := f.resolveNode(scale)
	off, _ := f.resolveNode(offset)
	s := f.stream()

	rank := x.shape.Rank()
	var reduceAxes []int
	for i := 0; i < rank; i++ {
		if i != featureAxis {
			reduceAxes = append(reduceAxes, i)
		}
	}

	mean := bridge.Sum(x.array, reduceAxes, true, s)
	countVal := float32(x.shape.Size() / x.shape.Dimensions[featureAxis])
	count := bridge.NewArrayScalarFloat32(countVal)
	meanDiv := bridge.Divide(mean, count, s)

	diff := bridge.Subtract(x.array, meanDiv, s)
	diffSq := bridge.Multiply(diff, diff, s)
	variance := bridge.Sum(diffSq, reduceAxes, true, s)
	varianceDiv := bridge.Divide(variance, count, s)

	eps := bridge.NewArrayScalarFloat32(epsilon)
	varPlusEps := bridge.Add(varianceDiv, eps, s)
	invStd := bridge.Rsqrt(varPlusEps, s)
	norm := bridge.Multiply(diff, invStd, s)
	scaled := bridge.Multiply(norm, sc.array, s)
	result := bridge.Add(scaled, off.array, s)

	meanOut := bridge.Squeeze(meanDiv, reduceAxes, s)
	varOut := bridge.Squeeze(varianceDiv, reduceAxes, s)

	featureDim := x.shape.Dimensions[featureAxis]
	statsShape := shapes.Make(x.shape.DType, featureDim)

	mean.Free()
	count.Free()
	diffSq.Free()
	variance.Free()
	eps.Free()
	varPlusEps.Free()
	invStd.Free()
	norm.Free()
	scaled.Free()

	xi, si, oi := x.tapeIdx, sc.tapeIdx, off.tapeIdx
	rAxes := append([]int{}, reduceAxes...)
	ep := epsilon
	cv := countVal

	// For BatchNormForTraining we need 3 tape entries for 3 outputs.
	// We compute all 3 in the first tape entry and cache them.
	normNode := f.record(x.shape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		cn := bridge.NewArrayScalarFloat32(cv)
		mn := bridge.Sum(arrays[xi], rAxes, true, s)
		md := bridge.Divide(mn, cn, s)
		df := bridge.Subtract(arrays[xi], md, s)
		ds := bridge.Multiply(df, df, s)
		vr := bridge.Sum(ds, rAxes, true, s)
		vd := bridge.Divide(vr, cn, s)
		e := bridge.NewArrayScalarFloat32(ep)
		vpe := bridge.Add(vd, e, s)
		is := bridge.Rsqrt(vpe, s)
		nm := bridge.Multiply(df, is, s)
		sc := bridge.Multiply(nm, arrays[si], s)
		res := bridge.Add(sc, arrays[oi], s)
		cn.Free()
		mn.Free()
		md.Free()
		df.Free()
		ds.Free()
		vr.Free()
		vd.Free()
		e.Free()
		vpe.Free()
		is.Free()
		nm.Free()
		sc.Free()
		return res
	})
	meanNode := f.record(statsShape, meanOut, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		cn := bridge.NewArrayScalarFloat32(cv)
		mn := bridge.Sum(arrays[xi], rAxes, true, s)
		md := bridge.Divide(mn, cn, s)
		res := bridge.Squeeze(md, rAxes, s)
		cn.Free()
		mn.Free()
		md.Free()
		return res
	})
	varNode := f.record(statsShape, varOut, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		cn := bridge.NewArrayScalarFloat32(cv)
		mn := bridge.Sum(arrays[xi], rAxes, true, s)
		md := bridge.Divide(mn, cn, s)
		df := bridge.Subtract(arrays[xi], md, s)
		ds := bridge.Multiply(df, df, s)
		vr := bridge.Sum(ds, rAxes, true, s)
		vd := bridge.Divide(vr, cn, s)
		res := bridge.Squeeze(vd, rAxes, s)
		cn.Free()
		mn.Free()
		md.Free()
		df.Free()
		ds.Free()
		vr.Free()
		vd.Free()
		return res
	})

	return normNode, meanNode, varNode, nil
}

func (f *Function) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Value,
	epsilon float32, featureAxis int) (gradOperand, gradScale, gradOffset backends.Value, err error) {
	// For now, return not implemented — GoMLX computes this via autograd.
	return nil, nil, nil, errors.Wrapf(backends.ErrNotImplemented, "BatchNormGradient")
}

// ===========================================================================
// RNG
// ===========================================================================

func (f *Function) RNGBitGenerator(state backends.Value, shape shapes.Shape) (newState, values backends.Value, err error) {
	f.markGoCallback("RNGBitGenerator")
	stateNode, _ := f.resolveNode(state)
	s := f.stream()

	if err := bridge.Eval(stateNode.array); err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: eval state")
	}

	seed := uint64(readScalarInt(stateNode.array, stateNode.shape.DType))
	key := bridge.RandomKey(seed)
	defer key.Free()

	bits := int(shape.DType.Size()) * 8
	var r *bridge.Array
	if bits > 4 {
		// MLX random_bits only supports width {1,2,4}. For wider types (e.g. Uint32),
		// generate Float32 uniform [0, 2^bits) and convert to the target integer dtype.
		lo := bridge.NewArrayScalarFloat32(0)
		hi := bridge.NewArrayScalarFloat32(float32(uint64(1) << bits))
		rf := bridge.RandomUniform(lo, hi, shape.Dimensions, bridge.DTypeFloat32, key, s)
		lo.Free()
		hi.Free()
		r = bridge.AsType(rf, gomlxDTypeToMLX(shape.DType), s)
		rf.Free()
	} else {
		r = bridge.RandomBits(shape.Dimensions, bits, key, s)
	}

	newSeed := seed + 1
	newStateArr := newScalarSeed(newSeed, stateNode.shape.DType)
	// The RNG state may be multi-element (e.g. [3] for GoMLX's Philox state).
	// Broadcast the scalar seed into the correct shape.
	newStateReshaped := bridge.BroadcastTo(newStateArr, stateNode.shape.Dimensions, s)
	newStateArr.Free()

	si := stateNode.tapeIdx
	stShape := stateNode.shape
	stGomlxDType := stateNode.shape.DType
	outShape := shape
	replayBits := bits
	replayDType := gomlxDTypeToMLX(shape.DType)
	stDims := append([]int{}, stateNode.shape.Dimensions...)
	oDims := append([]int{}, shape.Dimensions...)

	newStateNode := f.record(stShape, newStateReshaped, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		bridge.Eval(arrays[si])
		sd := uint64(readScalarInt(arrays[si], stGomlxDType))
		ns := newScalarSeed(sd+1, stGomlxDType)
		res := bridge.BroadcastTo(ns, stDims, s)
		ns.Free()
		return res
	})
	valNode := f.record(outShape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		bridge.Eval(arrays[si])
		sd := uint64(readScalarInt(arrays[si], stGomlxDType))
		k := bridge.RandomKey(sd)
		var res *bridge.Array
		if replayBits > 4 {
			lo := bridge.NewArrayScalarFloat32(0)
			hi := bridge.NewArrayScalarFloat32(float32(uint64(1) << replayBits))
			rf := bridge.RandomUniform(lo, hi, oDims, bridge.DTypeFloat32, k, s)
			lo.Free()
			hi.Free()
			res = bridge.AsType(rf, replayDType, s)
			rf.Free()
		} else {
			res = bridge.RandomBits(oDims, replayBits, k, s)
		}
		k.Free()
		return res
	})

	return newStateNode, valNode, nil
}

// ===========================================================================
// Pooling (ReduceWindow)
// ===========================================================================

// reduceWindowIdentity returns the identity element for a reduction type as a scalar array.
func reduceWindowIdentity(rt backends.ReduceOpType) *bridge.Array {
	switch rt {
	case backends.ReduceOpMax:
		return bridge.NewArrayScalarFloat32(-math.MaxFloat32)
	case backends.ReduceOpMin:
		return bridge.NewArrayScalarFloat32(math.MaxFloat32)
	case backends.ReduceOpProduct:
		return bridge.NewArrayScalarFloat32(1)
	default: // ReduceOpSum and anything else
		return bridge.NewArrayScalarFloat32(0)
	}
}

// applyReduceWindowPadding pads arr using the given paddings and returns the padded array
// and its updated dimensions. If no padding is needed, arr and inputDims are returned unchanged.
func applyReduceWindowPadding(arr *bridge.Array, inputDims []int, paddings [][2]int, rt backends.ReduceOpType, dt dtypes.DType, s *bridge.Stream) (*bridge.Array, []int) {
	needsPad := false
	for _, p := range paddings {
		if p[0] != 0 || p[1] != 0 {
			needsPad = true
			break
		}
	}
	if !needsPad {
		return arr, inputDims
	}

	axes := make([]int, len(paddings))
	lowPads := make([]int, len(paddings))
	highPads := make([]int, len(paddings))
	for i, p := range paddings {
		axes[i] = i
		lowPads[i] = p[0]
		highPads[i] = p[1]
	}
	padVal := reduceWindowIdentity(rt)
	if dt != dtypes.Float32 {
		padVal2 := bridge.AsType(padVal, gomlxDTypeToMLX(dt), s)
		padVal.Free()
		padVal = padVal2
	}
	padded := bridge.Pad(arr, padVal, axes, lowPads, highPads, s)
	padVal.Free()

	paddedDims := make([]int, len(inputDims))
	for i, d := range inputDims {
		if i < len(paddings) {
			paddedDims[i] = d + paddings[i][0] + paddings[i][1]
		} else {
			paddedDims[i] = d
		}
	}
	return padded, paddedDims
}

// applyReduceWindowStrided applies as_strided and then reduces over window axes.
func applyReduceWindowStrided(arr *bridge.Array, inputDims, windowDimensions, strides []int, rt backends.ReduceOpType, s *bridge.Stream) (*bridge.Array, []int) {
	rank := len(inputDims)
	outDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		outDims[i] = (inputDims[i]-windowDimensions[i])/strides[i] + 1
	}

	// Build strided shape: [outDim0, outDim1, ..., winDim0, winDim1, ...]
	stridedShape := make([]int, 2*rank)
	copy(stridedShape, outDims)
	copy(stridedShape[rank:], windowDimensions)

	// Compute element strides then build as_strided strides.
	elemStrides := make([]int64, rank)
	elemStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		elemStrides[i] = elemStrides[i+1] * int64(inputDims[i+1])
	}
	stridedStrides := make([]int64, 2*rank)
	for i := 0; i < rank; i++ {
		stridedStrides[i] = elemStrides[i] * int64(strides[i]) // output dim stride
		stridedStrides[rank+i] = elemStrides[i]                // window dim stride
	}

	windowed := bridge.AsStrided(arr, stridedShape, stridedStrides, 0, s)

	// Reduce over window dimensions (axes rank..2*rank-1).
	windowAxes := make([]int, rank)
	for i := 0; i < rank; i++ {
		windowAxes[i] = rank + i
	}
	switch rt {
	case backends.ReduceOpSum:
		return bridge.Sum(windowed, windowAxes, false, s), outDims
	case backends.ReduceOpMax:
		return bridge.Max(windowed, windowAxes, false, s), outDims
	case backends.ReduceOpMin:
		return bridge.Min(windowed, windowAxes, false, s), outDims
	case backends.ReduceOpProduct:
		return bridge.Prod(windowed, windowAxes, false, s), outDims
	}
	return nil, outDims
}

func (f *Function) ReduceWindow(
	x backends.Value,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Value, error) {
	f.markGoCallback("ReduceWindow") // compound op using AsStrided
	n, _ := f.resolveNode(x)
	s := f.stream()

	// Check for unsupported features.
	for _, d := range baseDilations {
		if d > 1 {
			return nil, errors.Errorf("ReduceWindow: base dilation > 1 not supported")
		}
	}
	for _, d := range windowDilations {
		if d > 1 {
			return nil, errors.Errorf("ReduceWindow: window dilation > 1 not supported")
		}
	}

	padded, paddedDims := applyReduceWindowPadding(n.array, n.shape.Dimensions, paddings, reductionType, n.shape.DType, s)
	result, outDims := applyReduceWindowStrided(padded, paddedDims, windowDimensions, strides, reductionType, s)
	if result == nil {
		return nil, errors.Errorf("ReduceWindow: unsupported reduction type %v", reductionType)
	}

	outShape := shapes.Make(n.shape.DType, outDims...)
	xi := n.tapeIdx
	rt := reductionType
	dt := n.shape.DType
	origDims := n.shape.Dimensions
	wd := append([]int{}, windowDimensions...)
	st := append([]int{}, strides...)
	pds := make([][2]int, len(paddings))
	copy(pds, paddings)
	return f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		padded, paddedDims := applyReduceWindowPadding(arrays[xi], origDims, pds, rt, dt, s)
		res, _ := applyReduceWindowStrided(padded, paddedDims, wd, st, rt, s)
		if padded != arrays[xi] {
			padded.Free()
		}
		return res
	}), nil
}

func (f *Function) SelectAndScatterMax(operand, source backends.Value,
	windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "SelectAndScatterMax: not yet implemented for MLX backend")
}

// ===========================================================================
// Fused Operations
// ===========================================================================

func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	r := bridge.Softmax(n.array, axis, true, f.stream())
	xi, ax := n.tapeIdx, axis
	node := f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return bridge.Softmax(arrays[xi], ax, true, s)
	})
	f.emitInstr(opSoftmax, int32(node.tapeIdx), []int32{int32(xi)}, int32(ax), 1) // precise=true
	return node, nil
}

func (f *Function) FusedGelu(x backends.Value, exact bool) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	s := f.stream()
	xi := n.tapeIdx
	dt := n.shape.DType
	if exact {
		sqrt2 := f.makeScalarConst(math.Sqrt2, dt)
		half := f.makeScalarConst(0.5, dt)
		one := f.makeScalarConst(1.0, dt)
		xDivSqrt2 := bridge.Divide(n.array, sqrt2, s)
		erfVal := bridge.Erf(xDivSqrt2, s)
		onePlusErf := bridge.Add(one, erfVal, s)
		halfTimesOPE := bridge.Multiply(half, onePlusErf, s)
		result := bridge.Multiply(n.array, halfTimesOPE, s)
		sqrt2.Free()
		half.Free()
		one.Free()
		xDivSqrt2.Free()
		erfVal.Free()
		onePlusErf.Free()
		halfTimesOPE.Free()
		node := f.record(n.shape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			sq2 := bridge.NewArrayScalarFloat32(float32(math.Sqrt2))
			h := bridge.NewArrayScalarFloat32(0.5)
			o := bridge.NewArrayScalarFloat32(1.0)
			xds := bridge.Divide(arrays[xi], sq2, s)
			ev := bridge.Erf(xds, s)
			ope := bridge.Add(o, ev, s)
			htope := bridge.Multiply(h, ope, s)
			res := bridge.Multiply(arrays[xi], htope, s)
			sq2.Free()
			h.Free()
			o.Free()
			xds.Free()
			ev.Free()
			ope.Free()
			htope.Free()
			return res
		})
		// C-tape: x * 0.5 * (1 + erf(x / sqrt(2)))
		sq2 := f.nextTempSlot()
		hf := f.nextTempSlot()
		one2 := f.nextTempSlot()
		xds := f.nextTempSlot()
		ev := f.nextTempSlot()
		ope := f.nextTempSlot()
		htope := f.nextTempSlot()
		f.emitInstr(opNewScalarF32, sq2, nil, float32Bits(float32(math.Sqrt2)))
		f.emitInstr(opNewScalarF32, hf, nil, float32Bits(0.5))
		f.emitInstr(opNewScalarF32, one2, nil, float32Bits(1.0))
		f.emitInstr(opDiv, xds, []int32{int32(xi), sq2})
		f.emitInstr(opErf, ev, []int32{xds})
		f.emitInstr(opAdd, ope, []int32{one2, ev})
		f.emitInstr(opMul, htope, []int32{hf, ope})
		f.emitInstr(opMul, int32(node.tapeIdx), []int32{int32(xi), htope})
		f.emitInstr(opFreeTemp, -1, []int32{sq2, hf, one2, xds, ev, ope, htope})
		return node, nil
	}
	c := f.makeScalarConst(0.044715, dt)
	sqrt2pi := f.makeScalarConst(math.Sqrt(2.0/math.Pi), dt)
	half := f.makeScalarConst(0.5, dt)
	one := f.makeScalarConst(1.0, dt)
	x3 := bridge.Power(n.array, f.makeScalarConst(3.0, dt), s)
	cx3 := bridge.Multiply(c, x3, s)
	xPlusCx3 := bridge.Add(n.array, cx3, s)
	inner := bridge.Multiply(sqrt2pi, xPlusCx3, s)
	tanhVal := bridge.Tanh(inner, s)
	onePlusTanh := bridge.Add(one, tanhVal, s)
	halfTimesOPT := bridge.Multiply(half, onePlusTanh, s)
	result := bridge.Multiply(n.array, halfTimesOPT, s)
	c.Free()
	sqrt2pi.Free()
	half.Free()
	one.Free()
	x3.Free()
	cx3.Free()
	xPlusCx3.Free()
	inner.Free()
	tanhVal.Free()
	onePlusTanh.Free()
	halfTimesOPT.Free()
	node := f.record(n.shape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		cc := bridge.NewArrayScalarFloat32(0.044715)
		sp := bridge.NewArrayScalarFloat32(float32(math.Sqrt(2.0 / math.Pi)))
		h := bridge.NewArrayScalarFloat32(0.5)
		o := bridge.NewArrayScalarFloat32(1.0)
		three := bridge.NewArrayScalarFloat32(3.0)
		xx3 := bridge.Power(arrays[xi], three, s)
		ccx3 := bridge.Multiply(cc, xx3, s)
		xpcx3 := bridge.Add(arrays[xi], ccx3, s)
		inn := bridge.Multiply(sp, xpcx3, s)
		tv := bridge.Tanh(inn, s)
		opt := bridge.Add(o, tv, s)
		htopt := bridge.Multiply(h, opt, s)
		res := bridge.Multiply(arrays[xi], htopt, s)
		cc.Free()
		sp.Free()
		h.Free()
		o.Free()
		three.Free()
		xx3.Free()
		ccx3.Free()
		xpcx3.Free()
		inn.Free()
		tv.Free()
		opt.Free()
		htopt.Free()
		return res
	})
	// C-tape: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	cc := f.nextTempSlot()
	sp := f.nextTempSlot()
	hf := f.nextTempSlot()
	one2 := f.nextTempSlot()
	three := f.nextTempSlot()
	xx3 := f.nextTempSlot()
	ccx3 := f.nextTempSlot()
	xpcx3 := f.nextTempSlot()
	inn := f.nextTempSlot()
	tv := f.nextTempSlot()
	opt := f.nextTempSlot()
	htopt := f.nextTempSlot()
	f.emitInstr(opNewScalarF32, cc, nil, float32Bits(0.044715))
	f.emitInstr(opNewScalarF32, sp, nil, float32Bits(float32(math.Sqrt(2.0/math.Pi))))
	f.emitInstr(opNewScalarF32, hf, nil, float32Bits(0.5))
	f.emitInstr(opNewScalarF32, one2, nil, float32Bits(1.0))
	f.emitInstr(opNewScalarF32, three, nil, float32Bits(3.0))
	f.emitInstr(opPow, xx3, []int32{int32(xi), three})
	f.emitInstr(opMul, ccx3, []int32{cc, xx3})
	f.emitInstr(opAdd, xpcx3, []int32{int32(xi), ccx3})
	f.emitInstr(opMul, inn, []int32{sp, xpcx3})
	f.emitInstr(opTanh, tv, []int32{inn})
	f.emitInstr(opAdd, opt, []int32{one2, tv})
	f.emitInstr(opMul, htopt, []int32{hf, opt})
	f.emitInstr(opMul, int32(node.tapeIdx), []int32{int32(xi), htopt})
	f.emitInstr(opFreeTemp, -1, []int32{cc, sp, hf, one2, three, xx3, ccx3, xpcx3, inn, tv, opt, htopt})
	return node, nil
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	n, _ := f.resolveNode(x)
	s := f.stream()

	xi := n.tapeIdx
	gi, bi := -1, -1
	var gArr, bArr *bridge.Array
	if gamma != nil {
		g, _ := f.resolveNode(gamma)
		gArr = g.array
		gi = g.tapeIdx
	}
	if beta != nil {
		b, _ := f.resolveNode(beta)
		bArr = b.array
		bi = b.tapeIdx
	}

	ax := append([]int{}, axes...)
	ep := float32(epsilon)

	if len(axes) == 1 && axes[0] == n.shape.Rank()-1 {
		r := bridge.FastLayerNorm(n.array, gArr, bArr, ep, s)
		node := f.record(n.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
			var ga, ba *bridge.Array
			if gi >= 0 {
				ga = arrays[gi]
			}
			if bi >= 0 {
				ba = arrays[bi]
			}
			return bridge.FastLayerNorm(arrays[xi], ga, ba, ep, s)
		})
		ins := []int32{int32(xi)}
		hasGamma := int32(0)
		hasBeta := int32(0)
		if gi >= 0 {
			ins = append(ins, int32(gi))
			hasGamma = 1
		}
		if bi >= 0 {
			ins = append(ins, int32(bi))
			hasBeta = 1
		}
		f.emitInstr(opFastLayerNorm, int32(node.tapeIdx), ins, float32Bits(ep), hasGamma, hasBeta)
		return node, nil
	}

	count := 1
	for _, a := range axes {
		count *= n.shape.Dimensions[a]
	}
	countArr := bridge.NewArrayScalarFloat32(float32(count))
	mean := bridge.Sum(n.array, axes, true, s)
	meanDiv := bridge.Divide(mean, countArr, s)
	diff := bridge.Subtract(n.array, meanDiv, s)
	diffSq := bridge.Multiply(diff, diff, s)
	variance := bridge.Sum(diffSq, axes, true, s)
	varianceDiv := bridge.Divide(variance, countArr, s)
	eps := bridge.NewArrayScalarFloat32(ep)
	varPlusEps := bridge.Add(varianceDiv, eps, s)
	invStd := bridge.Rsqrt(varPlusEps, s)
	result := bridge.Multiply(diff, invStd, s)

	if gArr != nil {
		result2 := bridge.Multiply(result, gArr, s)
		result.Free()
		result = result2
	}
	if bArr != nil {
		result2 := bridge.Add(result, bArr, s)
		result.Free()
		result = result2
	}

	mean.Free()
	countArr.Free()
	meanDiv.Free()
	diff.Free()
	diffSq.Free()
	variance.Free()
	varianceDiv.Free()
	eps.Free()
	varPlusEps.Free()
	invStd.Free()

	cv := float32(count)
	node := f.record(n.shape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		ca := bridge.NewArrayScalarFloat32(cv)
		mn := bridge.Sum(arrays[xi], ax, true, s)
		md := bridge.Divide(mn, ca, s)
		df := bridge.Subtract(arrays[xi], md, s)
		ds := bridge.Multiply(df, df, s)
		vr := bridge.Sum(ds, ax, true, s)
		vd := bridge.Divide(vr, ca, s)
		e := bridge.NewArrayScalarFloat32(ep)
		vpe := bridge.Add(vd, e, s)
		is := bridge.Rsqrt(vpe, s)
		res := bridge.Multiply(df, is, s)
		ca.Free()
		mn.Free()
		md.Free()
		df.Free()
		ds.Free()
		vr.Free()
		vd.Free()
		e.Free()
		vpe.Free()
		is.Free()
		if gi >= 0 {
			old := res
			res = bridge.Multiply(res, arrays[gi], s)
			old.Free()
		}
		if bi >= 0 {
			old := res
			res = bridge.Add(res, arrays[bi], s)
			old.Free()
		}
		return res
	})

	// C-tape: manual layer norm decomposition.
	// mean = sum(x, axes, keepdims) / count
	// diff = x - mean
	// var = sum(diff^2, axes, keepdims) / count
	// result = diff * rsqrt(var + eps)
	// optionally: result * gamma + beta
	var temps []int32
	caSlot := f.nextTempSlot()
	mnSlot := f.nextTempSlot()
	mdSlot := f.nextTempSlot()
	dfSlot := f.nextTempSlot()
	dsSlot := f.nextTempSlot()
	vrSlot := f.nextTempSlot()
	vdSlot := f.nextTempSlot()
	eSlot := f.nextTempSlot()
	vpeSlot := f.nextTempSlot()
	isSlot := f.nextTempSlot()
	temps = append(temps, caSlot, mnSlot, mdSlot, dfSlot, dsSlot, vrSlot, vdSlot, eSlot, vpeSlot, isSlot)

	// Encode axes as params: [num_axes, axes..., keepDims=1]
	axParams := []int32{int32(len(ax))}
	for _, a := range ax {
		axParams = append(axParams, int32(a))
	}
	axParams = append(axParams, 1) // keepDims = true

	f.emitInstr(opNewScalarF32, caSlot, nil, float32Bits(cv))
	f.emitInstr(opReduceSum, mnSlot, []int32{int32(xi)}, axParams...)
	f.emitInstr(opDiv, mdSlot, []int32{mnSlot, caSlot})
	f.emitInstr(opSub, dfSlot, []int32{int32(xi), mdSlot})
	f.emitInstr(opMul, dsSlot, []int32{dfSlot, dfSlot})
	f.emitInstr(opReduceSum, vrSlot, []int32{dsSlot}, axParams...)
	f.emitInstr(opDiv, vdSlot, []int32{vrSlot, caSlot})
	f.emitInstr(opNewScalarF32, eSlot, nil, float32Bits(ep))
	f.emitInstr(opAdd, vpeSlot, []int32{vdSlot, eSlot})
	f.emitInstr(opRsqrt, isSlot, []int32{vpeSlot})

	// Multiply diff * invStd → either output or another temp.
	normSlot := int32(node.tapeIdx)
	if gi >= 0 || bi >= 0 {
		normSlot = f.nextTempSlot()
		temps = append(temps, normSlot)
	}
	f.emitInstr(opMul, normSlot, []int32{dfSlot, isSlot})

	if gi >= 0 {
		scaledSlot := int32(node.tapeIdx)
		if bi >= 0 {
			scaledSlot = f.nextTempSlot()
			temps = append(temps, scaledSlot)
		}
		f.emitInstr(opMul, scaledSlot, []int32{normSlot, int32(gi)})
		normSlot = scaledSlot
	}
	if bi >= 0 {
		f.emitInstr(opAdd, int32(node.tapeIdx), []int32{normSlot, int32(bi)})
	}

	f.emitInstr(opFreeTemp, -1, temps)
	return node, nil
}

func (f *Function) FusedDense(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	xn, _ := f.resolveNode(x)
	wn, _ := f.resolveNode(weight)
	s := f.stream()
	xi, wi := xn.tapeIdx, wn.tapeIdx
	biasTapeIdx := -1

	result := bridge.MatMul(xn.array, wn.array, s)

	if bias != nil {
		bn, _ := f.resolveNode(bias)
		biasTapeIdx = bn.tapeIdx
		result2 := bridge.Add(result, bn.array, s)
		result.Free()
		result = result2
	}

	act := activation
	switch activation {
	case backends.ActivationGelu:
		// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
		sqrt2 := bridge.NewArrayScalarFloat32(float32(math.Sqrt2))
		half := bridge.NewArrayScalarFloat32(0.5)
		one := bridge.NewArrayScalarFloat32(1.0)
		xds := bridge.Divide(result, sqrt2, s)
		ev := bridge.Erf(xds, s)
		ope := bridge.Add(one, ev, s)
		htope := bridge.Multiply(half, ope, s)
		geluResult := bridge.Multiply(result, htope, s)
		sqrt2.Free()
		half.Free()
		one.Free()
		xds.Free()
		ev.Free()
		ope.Free()
		htope.Free()
		result.Free()
		result = geluResult
	case backends.ActivationRelu:
		zero := bridge.NewArrayScalarFloat32(0)
		reluResult := bridge.Maximum(result, zero, s)
		zero.Free()
		result.Free()
		result = reluResult
	case backends.ActivationSilu:
		sig := bridge.Sigmoid(result, s)
		siluResult := bridge.Multiply(result, sig, s)
		sig.Free()
		result.Free()
		result = siluResult
	case backends.ActivationTanh:
		tanhResult := bridge.Tanh(result, s)
		result.Free()
		result = tanhResult
	}

	outShape := dotGeneralShape(xn.shape, []int{xn.shape.Rank() - 1}, nil, wn.shape, []int{0}, nil)
	bi := biasTapeIdx
	node := f.record(outShape, result, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		res := bridge.MatMul(arrays[xi], arrays[wi], s)
		if bi >= 0 {
			old := res
			res = bridge.Add(res, arrays[bi], s)
			old.Free()
		}
		switch act {
		case backends.ActivationGelu:
			sq2 := bridge.NewArrayScalarFloat32(float32(math.Sqrt2))
			h := bridge.NewArrayScalarFloat32(0.5)
			o := bridge.NewArrayScalarFloat32(1.0)
			xds := bridge.Divide(res, sq2, s)
			ev := bridge.Erf(xds, s)
			ope := bridge.Add(o, ev, s)
			htope := bridge.Multiply(h, ope, s)
			old := res
			res = bridge.Multiply(res, htope, s)
			old.Free()
			sq2.Free()
			h.Free()
			o.Free()
			xds.Free()
			ev.Free()
			ope.Free()
			htope.Free()
		case backends.ActivationRelu:
			z := bridge.NewArrayScalarFloat32(0)
			old := res
			res = bridge.Maximum(res, z, s)
			old.Free()
			z.Free()
		case backends.ActivationSilu:
			sg := bridge.Sigmoid(res, s)
			old := res
			res = bridge.Multiply(res, sg, s)
			old.Free()
			sg.Free()
		case backends.ActivationTanh:
			old := res
			res = bridge.Tanh(res, s)
			old.Free()
		}
		return res
	})

	// Emit C-tape instructions: matmul + optional bias add + activation.
	outSlot := int32(node.tapeIdx)
	var temps []int32

	// Step 1: MatMul → temp or output slot.
	mmSlot := outSlot
	needTemp := (bi >= 0) || (act != backends.ActivationNone)
	if needTemp {
		mmSlot = f.nextTempSlot()
		temps = append(temps, mmSlot)
	}
	f.emitInstr(opMatMul, mmSlot, []int32{int32(xi), int32(wi)})

	// Step 2: Bias add → temp or output slot.
	addSlot := mmSlot
	if bi >= 0 {
		addSlot = outSlot
		if act != backends.ActivationNone {
			addSlot = f.nextTempSlot()
			temps = append(temps, addSlot)
		}
		f.emitInstr(opAdd, addSlot, []int32{mmSlot, int32(bi)})
	}

	// Step 3: Activation → output slot.
	switch act {
	case backends.ActivationRelu:
		// max(x, 0): create scalar zero, max, free zero
		zeroSlot := f.nextTempSlot()
		temps = append(temps, zeroSlot)
		f.emitInstr(opNewScalarF32, zeroSlot, nil, float32Bits(0))
		f.emitInstr(opMax, outSlot, []int32{addSlot, zeroSlot})
	case backends.ActivationGelu:
		// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
		sq2 := f.nextTempSlot()
		hf := f.nextTempSlot()
		one := f.nextTempSlot()
		xds := f.nextTempSlot()
		ev := f.nextTempSlot()
		ope := f.nextTempSlot()
		htope := f.nextTempSlot()
		temps = append(temps, sq2, hf, one, xds, ev, ope, htope)
		f.emitInstr(opNewScalarF32, sq2, nil, float32Bits(float32(math.Sqrt2)))
		f.emitInstr(opNewScalarF32, hf, nil, float32Bits(0.5))
		f.emitInstr(opNewScalarF32, one, nil, float32Bits(1.0))
		f.emitInstr(opDiv, xds, []int32{addSlot, sq2})
		f.emitInstr(opErf, ev, []int32{xds})
		f.emitInstr(opAdd, ope, []int32{one, ev})
		f.emitInstr(opMul, htope, []int32{hf, ope})
		f.emitInstr(opMul, outSlot, []int32{addSlot, htope})
	case backends.ActivationSilu:
		// SiLU: x * sigmoid(x)
		sigSlot := f.nextTempSlot()
		temps = append(temps, sigSlot)
		f.emitInstr(opSigmoid, sigSlot, []int32{addSlot})
		f.emitInstr(opMul, outSlot, []int32{addSlot, sigSlot})
	case backends.ActivationTanh:
		f.emitInstr(opTanh, outSlot, []int32{addSlot})
	case backends.ActivationNone:
		// No activation — the last step already wrote to outSlot.
	}

	if len(temps) > 0 {
		f.emitInstr(opFreeTemp, -1, temps)
	}

	return node, nil
}

func (f *Function) FusedScaledDotProductAttention(
	query, key, value, mask backends.Value,
	numHeads, numKVHeads int,
	axesLayout backends.AxesLayout,
	scale float64,
	causal bool,
) (backends.Value, error) {
	q, _ := f.resolveNode(query)
	k, _ := f.resolveNode(key)
	v, _ := f.resolveNode(value)
	s := f.stream()

	qi, ki, vi := q.tapeIdx, k.tapeIdx, v.tapeIdx
	mi := -1
	var maskArr *bridge.Array
	if mask != nil {
		m, _ := f.resolveNode(mask)
		maskArr = m.array
		mi = m.tapeIdx
	}

	sc := float32(scale)
	r := bridge.FastScaledDotProductAttention(q.array, k.array, v.array, maskArr, sc, s)
	node := f.record(q.shape, r, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		var ma *bridge.Array
		if mi >= 0 {
			ma = arrays[mi]
		}
		return bridge.FastScaledDotProductAttention(arrays[qi], arrays[ki], arrays[vi], ma, sc, s)
	})
	ins := []int32{int32(qi), int32(ki), int32(vi)}
	hasMask := int32(0)
	if mi >= 0 {
		ins = append(ins, int32(mi))
		hasMask = 1
	}
	f.emitInstr(opFastSDPA, int32(node.tapeIdx), ins, float32Bits(sc), hasMask)
	return node, nil
}

func (f *Function) FusedAttentionQKVProjection(
	x, wQKV, biasQ, biasK, biasV backends.Value,
	queryDim, keyValueDim int,
) (query, key, value backends.Value, err error) {
	f.markGoCallback("FusedAttentionQKVProjection") // compound op
	xn, _ := f.resolveNode(x)
	wn, _ := f.resolveNode(wQKV)
	s := f.stream()
	xi, wi := xn.tapeIdx, wn.tapeIdx
	bqi, bki, bvi := -1, -1, -1

	combined := bridge.MatMul(xn.array, wn.array, s)

	totalDim := queryDim + 2*keyValueDim
	batchDims := xn.shape.Dimensions[:len(xn.shape.Dimensions)-1]

	// sliceLastAxis slices [start, stop) on the last dimension of arr.
	sliceLastAxis := func(arr *bridge.Array, start, stop int) *bridge.Array {
		ndim := len(batchDims) + 1
		starts := make([]int, ndim)
		starts[ndim-1] = start
		stops := append(append([]int{}, batchDims...), stop)
		strides := make([]int, ndim)
		for i := range strides {
			strides[i] = 1
		}
		return bridge.Slice(arr, starts, stops, strides, s)
	}
	qArr := sliceLastAxis(combined, 0, queryDim)
	kArr := sliceLastAxis(combined, queryDim, queryDim+keyValueDim)
	vArr := sliceLastAxis(combined, queryDim+keyValueDim, totalDim)
	combined.Free()

	if biasQ != nil {
		bq, _ := f.resolveNode(biasQ)
		bqi = bq.tapeIdx
		qArr2 := bridge.Add(qArr, bq.array, s)
		qArr.Free()
		qArr = qArr2
	}
	if biasK != nil {
		bk, _ := f.resolveNode(biasK)
		bki = bk.tapeIdx
		kArr2 := bridge.Add(kArr, bk.array, s)
		kArr.Free()
		kArr = kArr2
	}
	if biasV != nil {
		bv, _ := f.resolveNode(biasV)
		bvi = bv.tapeIdx
		vArr2 := bridge.Add(vArr, bv.array, s)
		vArr.Free()
		vArr = vArr2
	}

	qDims := append(append([]int{}, batchDims...), queryDim)
	kDims := append(append([]int{}, batchDims...), keyValueDim)
	qShape := shapes.Make(xn.shape.DType, qDims...)
	kShape := shapes.Make(xn.shape.DType, kDims...)

	// sliceQKVReplay is the tape-replay equivalent of sliceLastAxis + optional bias add,
	// shared across the three Q/K/V tape closures.
	qd, kvd, td := queryDim, keyValueDim, totalDim
	bd := append([]int{}, batchDims...)
	sliceQKVReplay := func(arrays []*bridge.Array, s *bridge.Stream, start, stop, biasIdx int) *bridge.Array {
		comb := bridge.MatMul(arrays[xi], arrays[wi], s)
		ndim := len(bd) + 1
		starts := make([]int, ndim)
		starts[ndim-1] = start
		stops := append(append([]int{}, bd...), stop)
		strides := make([]int, ndim)
		for i := range strides {
			strides[i] = 1
		}
		res := bridge.Slice(comb, starts, stops, strides, s)
		comb.Free()
		if biasIdx >= 0 {
			old := res
			res = bridge.Add(res, arrays[biasIdx], s)
			old.Free()
		}
		return res
	}

	qNode := f.record(qShape, qArr, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return sliceQKVReplay(arrays, s, 0, qd, bqi)
	})
	kNode := f.record(kShape, kArr, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return sliceQKVReplay(arrays, s, qd, qd+kvd, bki)
	})
	vNode := f.record(kShape, vArr, func(arrays []*bridge.Array, s *bridge.Stream) *bridge.Array {
		return sliceQKVReplay(arrays, s, qd+kvd, td, bvi)
	})

	return qNode, kNode, vNode, nil
}

// ===========================================================================
// C Tape Serialization
// ===========================================================================

// serializeTape converts the instrMeta slice into a flat int32 instruction stream
// and related metadata for the C tape interpreter.
func (f *Function) serializeTape() (instrs []int32, numSlots int, outputSlots, constSlots []int32, constArrays []*bridge.Array, paramSlots []int32) {
	numSlots = len(f.tape) + f.tempSlotCount

	// Build output slots.
	outputSlots = make([]int32, len(f.outputs))
	for i, out := range f.outputs {
		outputSlots[i] = int32(out.tapeIdx)
	}

	// Build const slots and arrays.
	constSlots = make([]int32, len(f.constIndices))
	constArrays = make([]*bridge.Array, len(f.constIndices))
	for i, idx := range f.constIndices {
		constSlots[i] = int32(idx)
		// Get the constant array from the tape closure by replaying it.
		constArrays[i] = f.tape[idx](nil, nil)
	}

	// Build param slots.
	paramSlots = make([]int32, len(f.params))
	for i, p := range f.params {
		paramSlots[i] = int32(p.tapeIdx)
	}

	// remapSlot translates sentinel temp slot indices (tempSlotBase+N) into the
	// actual slot range [len(f.tape), len(f.tape)+tempSlotCount).
	tapeLen := int32(len(f.tape))
	remapSlot := func(slot int32) int32 {
		if slot >= int32(tempSlotBase) {
			return tapeLen + (slot - int32(tempSlotBase))
		}
		return slot
	}

	// Flatten instrMeta to contiguous int32 stream.
	// Each instruction: [opcode, out_slot, num_inputs, input_slots..., num_params, params...]
	for _, instr := range f.instrs {
		instrs = append(instrs, instr.opcode)
		instrs = append(instrs, remapSlot(instr.outSlot))
		instrs = append(instrs, int32(len(instr.inputs)))
		for _, inp := range instr.inputs {
			instrs = append(instrs, remapSlot(inp))
		}
		instrs = append(instrs, int32(len(instr.params)))
		instrs = append(instrs, instr.params...)
	}

	return
}

// ===========================================================================
// Control Flow
// ===========================================================================

// buildOutputNodes creates output graphNodes and backends.Value results for a
// control flow operation, given the output shapes.
func (f *Function) buildOutputNodes(outputShapes []shapes.Shape) ([]*graphNode, []backends.Value) {
	outputNodes := make([]*graphNode, len(outputShapes))
	results := make([]backends.Value, len(outputShapes))
	for i, s := range outputShapes {
		n := &graphNode{shape: s, owner: f}
		outputNodes[i] = n
		results[i] = n
	}
	return outputNodes, results
}

func (f *Function) While(cond, body backends.Function, initialState ...backends.Value) ([]backends.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("While: only one control flow operation per function is supported")
	}
	if len(initialState) == 0 {
		return nil, errors.New("While: requires at least one initial state value")
	}

	stateNodes, err := f.resolveNodes("While", initialState...)
	if err != nil {
		return nil, err
	}

	condFn, err := f.validateClosure("While", "cond", cond)
	if err != nil {
		return nil, err
	}
	bodyFn, err := f.validateClosure("While", "body", body)
	if err != nil {
		return nil, err
	}

	outputShapes := make([]shapes.Shape, len(stateNodes))
	for i, n := range stateNodes {
		outputShapes[i] = n.shape.Clone()
	}

	outputNodes, results := f.buildOutputNodes(outputShapes)

	f.controlFlowStep = &controlFlowStep{
		opType:       backends.OpTypeWhile,
		inputs:       stateNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		whileData:    &whileStepData{condFn: condFn, bodyFn: bodyFn},
	}

	return results, nil
}

func (f *Function) If(pred backends.Value, trueBranch, falseBranch backends.Function) ([]backends.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("If: only one control flow operation per function is supported")
	}

	predNode, err := f.resolveNode(pred)
	if err != nil {
		return nil, err
	}

	trueFn, err := f.validateClosure("If", "trueBranch", trueBranch)
	if err != nil {
		return nil, err
	}
	falseFn, err := f.validateClosure("If", "falseBranch", falseBranch)
	if err != nil {
		return nil, err
	}

	outputShapes := make([]shapes.Shape, len(trueFn.outputs))
	for i, out := range trueFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	outputNodes, results := f.buildOutputNodes(outputShapes)

	f.controlFlowStep = &controlFlowStep{
		opType:       backends.OpTypeIf,
		inputs:       []*graphNode{predNode},
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		ifData:       &ifStepData{trueFn: trueFn, falseFn: falseFn},
	}

	return results, nil
}

func (f *Function) Sort(comparator backends.Function, axis int, isStable bool, inputs ...backends.Value) ([]backends.Value, error) {
	if f.controlFlowStep != nil {
		return nil, errors.New("Sort: only one control flow operation per function is supported")
	}

	inputNodes, err := f.resolveNodes("Sort", inputs...)
	if err != nil {
		return nil, err
	}

	compFn, err := f.validateClosure("Sort", "comparator", comparator)
	if err != nil {
		return nil, err
	}

	outputShapes := make([]shapes.Shape, len(inputNodes))
	for i, n := range inputNodes {
		outputShapes[i] = n.shape.Clone()
	}

	outputNodes, results := f.buildOutputNodes(outputShapes)

	f.controlFlowStep = &controlFlowStep{
		opType:       backends.OpTypeSort,
		inputs:       inputNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		sortData:     &sortStepData{comparatorFn: compFn, axis: axis, isStable: isStable},
	}

	return results, nil
}

func (f *Function) Call(fn backends.Function, inputs ...backends.Value) ([]backends.Value, error) {
	inputNodes, err := f.resolveNodes("Call", inputs...)
	if err != nil {
		return nil, err
	}

	targetFn, ok := fn.(*Function)
	if !ok {
		return nil, errors.Errorf("Call: target must be *mlx.Function, got %T", fn)
	}

	outputShapes := make([]shapes.Shape, len(targetFn.outputs))
	for i, out := range targetFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	outputNodes, results := f.buildOutputNodes(outputShapes)

	f.controlFlowStep = &controlFlowStep{
		opType:       backends.OpTypeCall,
		inputs:       inputNodes,
		outputShapes: outputShapes,
		outputNodes:  outputNodes,
		callData:     &callStepData{targetFn: targetFn},
	}

	return results, nil
}

// Ensure all imports are used.
var (
	_ = math.Sqrt2
	_ = reflect.ValueOf
	_ = runtime.KeepAlive
	_ = unsafe.Pointer(nil)
	_ = bridge.Add
	_ = backends.OpTypeAdd
	_ = notimplemented.Function{}
	_ = shapeinference.BinaryOp
	_ = dtypes.Float32
	_ = shapes.Make
	_ = errors.New
)
