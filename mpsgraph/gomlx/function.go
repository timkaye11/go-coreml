// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package mpsgraph

import (
	"reflect"
	"unsafe"

	"github.com/gomlx/go-coreml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// graphNode represents a value in the computation graph.
type graphNode struct {
	tensor bridge.Tensor // MPSGraphTensor handle
	shape  shapes.Shape
	name   string // Optional name (for parameters)
}

// Function implements backends.Function for the MPSGraph backend.
type Function struct {
	notimplemented.Function // Bootstrap: unimplemented ops return ErrNotImplemented.

	builder  *Builder
	name     string
	parent   *Function
	returned bool
	params   []*graphNode // Input placeholders
	outputs  []*graphNode // Return values
}

// Verify interface compliance.
var _ backends.Function = &Function{}

// newFunction creates a new function in the computation graph.
func newFunction(builder *Builder, name string, parent *Function) *Function {
	return &Function{
		builder: builder,
		name:    name,
		parent:  parent,
	}
}

// Name returns the function name.
func (f *Function) Name() string { return f.name }

// Parent returns the parent function.
func (f *Function) Parent() backends.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// Closure creates a closure function.
func (f *Function) Closure() (backends.Function, error) {
	return newFunction(f.builder, "", f), nil
}

// ctx returns the bridge context for this function's builder.
func (f *Function) ctx() *bridge.Context { return f.builder.ctx }

// castNode converts a backends.Value to a graphNode.
func castNode(v backends.Value) (*graphNode, error) {
	n, ok := v.(*graphNode)
	if !ok {
		return nil, errors.Errorf("expected *graphNode, got %T", v)
	}
	return n, nil
}

// castNodes converts multiple backends.Value to graphNodes.
func castNodes(name string, values ...backends.Value) ([]*graphNode, error) {
	nodes := make([]*graphNode, len(values))
	for i, v := range values {
		n, err := castNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: input #%d", name, i)
		}
		nodes[i] = n
	}
	return nodes, nil
}

// ===========================================================================
// Lifecycle: Parameter, Constant, Return
// ===========================================================================

// Parameter creates an input placeholder.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Value, error) {
	dims := make([]int64, shape.Rank())
	for i, d := range shape.Dimensions {
		dims[i] = int64(d)
	}
	dtype := dtypeToBridgeDType(shape.DType)
	tensor, err := f.ctx().Placeholder(dtype, dims)
	if err != nil {
		return nil, errors.Wrapf(err, "Parameter(%s)", name)
	}
	node := &graphNode{tensor: tensor, shape: shape, name: name}
	f.params = append(f.params, node)
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

	bridgeDType := dtypeToBridgeDType(dt)

	var dataPtr unsafe.Pointer
	var nbytes int64
	if flatVal.Len() > 0 {
		dataPtr = unsafe.Pointer(flatVal.Pointer())
		nbytes = int64(flatVal.Len()) * int64(dt.Size())
	}

	// MPSGraph requires shape.count > 0 (no rank-0 tensors for constants).
	// For scalars, create as [1] and reshape to scalar.
	isScalar := len(dims) == 0
	shapeDims := make([]int64, len(dims))
	for i, d := range dims {
		shapeDims[i] = int64(d)
	}
	if isScalar {
		shapeDims = []int64{1}
	}

	tensor, err := f.ctx().Constant(dataPtr, nbytes, bridgeDType, shapeDims)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	if isScalar {
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, "Constant: reshape to scalar")
		}
	}

	return &graphNode{tensor: tensor, shape: shape}, nil
}

// Return marks the function outputs.
func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	nodes, err := castNodes("Return", outputs...)
	if err != nil {
		return err
	}
	f.outputs = nodes
	f.returned = true
	return nil
}

// Call calls another function with the given inputs.
func (f *Function) Call(fn backends.Function, inputs ...backends.Value) ([]backends.Value, error) {
	return nil, errors.Wrapf(notimplemented.NotImplementedError, "Call")
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func (f *Function) unaryOp(opName string, opType backends.OpType, bridgeFn func(bridge.Tensor) (bridge.Tensor, error), x backends.Value) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	outShape, err := shapeinference.UnaryOp(opType, node.shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Abs(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Abs", backends.OpTypeAbs, f.ctx().Abs, x)
}

func (f *Function) Neg(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Neg", backends.OpTypeNeg, f.ctx().Neg, x)
}

func (f *Function) Sqrt(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Sqrt", backends.OpTypeSqrt, f.ctx().Sqrt, x)
}

func (f *Function) Rsqrt(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Rsqrt", backends.OpTypeRsqrt, f.ctx().Rsqrt, x)
}

func (f *Function) Exp(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Exp", backends.OpTypeExp, f.ctx().Exp, x)
}

func (f *Function) Expm1(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Expm1", backends.OpTypeExpm1, f.ctx().Expm1, x)
}

func (f *Function) Log(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Log", backends.OpTypeLog, f.ctx().Log, x)
}

func (f *Function) Log1p(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Log1p", backends.OpTypeLog1p, f.ctx().Log1p, x)
}

func (f *Function) Sin(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Sin", backends.OpTypeSin, f.ctx().Sin, x)
}

func (f *Function) Cos(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Cos", backends.OpTypeCos, f.ctx().Cos, x)
}

func (f *Function) Tanh(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Tanh", backends.OpTypeTanh, f.ctx().Tanh, x)
}

func (f *Function) Logistic(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Logistic", backends.OpTypeLogistic, f.ctx().Sigmoid, x)
}

func (f *Function) Erf(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Erf", backends.OpTypeErf, f.ctx().Erf, x)
}

func (f *Function) Floor(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Floor", backends.OpTypeFloor, f.ctx().Floor, x)
}

func (f *Function) Ceil(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Ceil", backends.OpTypeCeil, f.ctx().Ceil, x)
}

func (f *Function) Round(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Round", backends.OpTypeRound, f.ctx().Round, x)
}

func (f *Function) Sign(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Sign", backends.OpTypeSign, f.ctx().Sign, x)
}

func (f *Function) LogicalNot(x backends.Value) (backends.Value, error) {
	return f.unaryOp("LogicalNot", backends.OpTypeLogicalNot, f.ctx().LogicalNot, x)
}

func (f *Function) BitwiseNot(x backends.Value) (backends.Value, error) {
	return f.unaryOp("BitwiseNot", backends.OpTypeBitwiseNot, f.ctx().BitwiseNot, x)
}

func (f *Function) IsFinite(x backends.Value) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	outShape := shapes.Make(dtypes.Bool, node.shape.Dimensions...)
	tensor, err := f.ctx().IsFinite(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "IsFinite")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) IsNaN(x backends.Value) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	outShape := shapes.Make(dtypes.Bool, node.shape.Dimensions...)
	tensor, err := f.ctx().IsNaN(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "IsNaN")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Identity(x backends.Value) (backends.Value, error) {
	return f.unaryOp("Identity", backends.OpTypeIdentity, f.ctx().Identity, x)
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func (f *Function) binaryOp(opName string, opType backends.OpType, bridgeFn func(bridge.Tensor, bridge.Tensor) (bridge.Tensor, error), lhs, rhs backends.Value) (backends.Value, error) {
	nodes, err := castNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.BinaryOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(nodes[0].tensor, nodes[1].tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Add(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Add", backends.OpTypeAdd, f.ctx().Add, lhs, rhs)
}

func (f *Function) Sub(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Sub", backends.OpTypeSub, f.ctx().Sub, lhs, rhs)
}

func (f *Function) Mul(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Mul", backends.OpTypeMul, f.ctx().Mul, lhs, rhs)
}

func (f *Function) Div(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Div", backends.OpTypeDiv, f.ctx().Div, lhs, rhs)
}

func (f *Function) Rem(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Rem", backends.OpTypeRem, f.ctx().Rem, lhs, rhs)
}

func (f *Function) Pow(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Pow", backends.OpTypePow, f.ctx().Pow, lhs, rhs)
}

func (f *Function) Max(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Max", backends.OpTypeMax, f.ctx().Max, lhs, rhs)
}

func (f *Function) Min(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Min", backends.OpTypeMin, f.ctx().Min, lhs, rhs)
}

func (f *Function) Atan2(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("Atan2", backends.OpTypeAtan2, f.ctx().Atan2, lhs, rhs)
}

func (f *Function) LogicalAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("LogicalAnd", backends.OpTypeLogicalAnd, f.ctx().LogicalAnd, lhs, rhs)
}

func (f *Function) LogicalOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("LogicalOr", backends.OpTypeLogicalOr, f.ctx().LogicalOr, lhs, rhs)
}

func (f *Function) LogicalXor(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("LogicalXor", backends.OpTypeLogicalXor, f.ctx().LogicalXor, lhs, rhs)
}

func (f *Function) BitwiseAnd(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseAnd", backends.OpTypeBitwiseAnd, f.ctx().BitwiseAnd, lhs, rhs)
}

func (f *Function) BitwiseOr(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseOr", backends.OpTypeBitwiseOr, f.ctx().BitwiseOr, lhs, rhs)
}

func (f *Function) BitwiseXor(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("BitwiseXor", backends.OpTypeBitwiseXor, f.ctx().BitwiseXor, lhs, rhs)
}

func (f *Function) ShiftLeft(lhs, rhs backends.Value) (backends.Value, error) {
	return f.binaryOp("ShiftLeft", backends.OpTypeShiftLeft, f.ctx().ShiftLeft, lhs, rhs)
}

func (f *Function) ShiftRightArithmetic(lhs, rhs backends.Value) (backends.Value, error) {
	// MPSGraph's shiftRight is arithmetic for signed types.
	return f.binaryOp("ShiftRightArithmetic", backends.OpTypeShiftRightArithmetic, f.ctx().ShiftRight, lhs, rhs)
}

func (f *Function) ShiftRightLogical(lhs, rhs backends.Value) (backends.Value, error) {
	// TODO: implement logical shift right (treat as unsigned).
	return f.binaryOp("ShiftRightLogical", backends.OpTypeShiftRightLogical, f.ctx().ShiftRight, lhs, rhs)
}

// --- Comparison ---

func (f *Function) Equal(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("Equal", backends.OpTypeEqual, f.ctx().Equal, lhs, rhs)
}

func (f *Function) NotEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("NotEqual", backends.OpTypeNotEqual, f.ctx().NotEqual, lhs, rhs)
}

func (f *Function) GreaterThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("GreaterThan", backends.OpTypeGreaterThan, f.ctx().GreaterThan, lhs, rhs)
}

func (f *Function) GreaterOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("GreaterOrEqual", backends.OpTypeGreaterOrEqual, f.ctx().GreaterOrEqual, lhs, rhs)
}

func (f *Function) LessThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("LessThan", backends.OpTypeLessThan, f.ctx().LessThan, lhs, rhs)
}

func (f *Function) LessOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.comparisonOp("LessOrEqual", backends.OpTypeLessOrEqual, f.ctx().LessOrEqual, lhs, rhs)
}

func (f *Function) comparisonOp(opName string, opType backends.OpType, bridgeFn func(bridge.Tensor, bridge.Tensor) (bridge.Tensor, error), lhs, rhs backends.Value) (backends.Value, error) {
	nodes, err := castNodes(opName, lhs, rhs)
	if err != nil {
		return nil, err
	}
	// Comparison output shape: broadcast shape of inputs, dtype Bool.
	outShape, err := shapeinference.ComparisonOp(opType, nodes[0].shape, nodes[1].shape)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := bridgeFn(nodes[0].tensor, nodes[1].tensor)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

// ===========================================================================
// Shape Operations
// ===========================================================================

func (f *Function) Reshape(x backends.Value, dimensions ...int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	outShape := shapes.Make(node.shape.DType, dimensions...)
	dims := make([]int64, len(dimensions))
	for i, d := range dimensions {
		dims[i] = int64(d)
	}
	tensor, err := f.ctx().Reshape(node.tensor, dims)
	if err != nil {
		return nil, errors.Wrap(err, "Reshape")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Transpose(x backends.Value, permutation ...int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	outShape, err := shapeinference.TransposeOp(node.shape, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	tensor, err := f.ctx().Transpose(node.tensor, permutation)
	if err != nil {
		return nil, errors.Wrap(err, "Transpose")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) ConvertDType(x backends.Value, dtype dtypes.DType) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	outShape := shapes.Make(dtype, node.shape.Dimensions...)
	bridgeDType := dtypeToBridgeDType(dtype)
	tensor, err := f.ctx().Cast(node.tensor, bridgeDType)
	if err != nil {
		return nil, errors.Wrap(err, "ConvertDType")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) BroadcastInDim(x backends.Value, outputShape shapes.Shape, broadcastAxes []int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim")
	}

	// BroadcastInDim: x's i-th axis maps to broadcastAxes[i]-th axis of output.
	// We need to reshape x by inserting size-1 dimensions, then broadcast.
	intermediateShape := make([]int64, outputShape.Rank())
	for i := range intermediateShape {
		intermediateShape[i] = 1
	}
	for i, outAxis := range broadcastAxes {
		intermediateShape[outAxis] = int64(node.shape.Dimensions[i])
	}

	// Reshape to intermediate (with 1s in non-broadcast dims).
	reshaped, err := f.ctx().Reshape(node.tensor, intermediateShape)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim: reshape")
	}

	// Broadcast to output shape.
	outDims := make([]int64, outputShape.Rank())
	for i, d := range outputShape.Dimensions {
		outDims[i] = int64(d)
	}
	tensor, err := f.ctx().BroadcastTo(reshaped, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "BroadcastInDim: broadcast")
	}
	return &graphNode{tensor: tensor, shape: outputShape}, nil
}

func (f *Function) Where(condition, onTrue, onFalse backends.Value) (backends.Value, error) {
	nodes, err := castNodes("Where", condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.WhereOp(nodes[0].shape, nodes[1].shape, nodes[2].shape)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	tensor, err := f.ctx().Where(nodes[0].tensor, nodes[1].tensor, nodes[2].tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Where")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Clamp(min, x, max backends.Value) (backends.Value, error) {
	nodes, err := castNodes("Clamp", min, x, max)
	if err != nil {
		return nil, err
	}
	// Output shape is the same as x's shape.
	outShape := nodes[1].shape
	tensor, err := f.ctx().Clamp(nodes[0].tensor, nodes[1].tensor, nodes[2].tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Clamp")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Slice(operand backends.Value, starts, limits, strides []int) (backends.Value, error) {
	node, err := castNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	outShape, err := shapeinference.SliceOp(node.shape, starts, limits, strides)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	starts64 := make([]int64, len(starts))
	ends64 := make([]int64, len(limits))
	strides64 := make([]int64, len(strides))
	for i := range starts {
		starts64[i] = int64(starts[i])
		ends64[i] = int64(limits[i])
		strides64[i] = int64(strides[i])
	}
	tensor, err := f.ctx().Slice(node.tensor, starts64, ends64, strides64)
	if err != nil {
		return nil, errors.Wrap(err, "Slice")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Concatenate(axis int, operands ...backends.Value) (backends.Value, error) {
	nodes, err := castNodes("Concatenate", operands...)
	if err != nil {
		return nil, err
	}
	inputShapes := make([]shapes.Shape, len(nodes))
	tensors := make([]bridge.Tensor, len(nodes))
	for i, n := range nodes {
		inputShapes[i] = n.shape
		tensors[i] = n.tensor
	}
	outShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	tensor, err := f.ctx().Concatenate(tensors, axis)
	if err != nil {
		return nil, errors.Wrap(err, "Concatenate")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) Reverse(x backends.Value, axes ...int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	tensor, err := f.ctx().Reverse(node.tensor, axes)
	if err != nil {
		return nil, errors.Wrap(err, "Reverse")
	}
	return &graphNode{tensor: tensor, shape: node.shape}, nil
}

func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (backends.Value, error) {
	dims := make([]int64, shape.Rank())
	for i, d := range shape.Dimensions {
		dims[i] = int64(d)
	}
	dtype := dtypeToBridgeDType(shape.DType)
	tensor, err := f.ctx().Iota(dtype, dims, iotaAxis)
	if err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	return &graphNode{tensor: tensor, shape: shape}, nil
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

func (f *Function) Dot(lhs, rhs backends.Value) (backends.Value, error) {
	nodes, err := castNodes("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsShape := nodes[0].shape
	rhsShape := nodes[1].shape

	// Dot product: [M,K] x [K,N] → [M,N], or [K] x [K] → scalar.
	var outShape shapes.Shape
	switch {
	case lhsShape.Rank() == 2 && rhsShape.Rank() == 2:
		if lhsShape.Dimensions[1] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: incompatible shapes %s and %s", lhsShape, rhsShape)
		}
		outShape = shapes.Make(lhsShape.DType, lhsShape.Dimensions[0], rhsShape.Dimensions[1])
	case lhsShape.Rank() == 1 && rhsShape.Rank() == 1:
		if lhsShape.Dimensions[0] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: incompatible shapes %s and %s", lhsShape, rhsShape)
		}
		outShape = shapes.Make(lhsShape.DType)
	default:
		return nil, errors.Errorf("Dot: unsupported rank combination %d and %d", lhsShape.Rank(), rhsShape.Rank())
	}

	tensor, err := f.ctx().MatMul(nodes[0].tensor, nodes[1].tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Dot")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

// DotGeneral is implemented in dotgeneral.go.

// ===========================================================================
// Reduction Operations
// ===========================================================================

func (f *Function) reduceOp(opName string, opType backends.OpType, reduceType int, x backends.Value, axes ...int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	outShape, err := shapeinference.ReduceOp(node.shape, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	tensor, err := f.ctx().Reduce(node.tensor, reduceType, axes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}
	// MPSGraph reductions keep reduced dims as size 1 — reshape to squeeze them.
	if outShape.Rank() > 0 {
		outDims := make([]int64, outShape.Rank())
		for i, d := range outShape.Dimensions {
			outDims[i] = int64(d)
		}
		tensor, err = f.ctx().Reshape(tensor, outDims)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape after reduce")
		}
	} else {
		// Scalar output: reshape to rank 0.
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape to scalar")
		}
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) ReduceSum(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp("ReduceSum", backends.OpTypeReduceSum, bridge.ReduceSum, x, axes...)
}

func (f *Function) ReduceMax(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp("ReduceMax", backends.OpTypeReduceMax, bridge.ReduceMax, x, axes...)
}

func (f *Function) ReduceMin(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp("ReduceMin", backends.OpTypeReduceMin, bridge.ReduceMin, x, axes...)
}

func (f *Function) ReduceProduct(x backends.Value, axes ...int) (backends.Value, error) {
	return f.reduceOp("ReduceProduct", backends.OpTypeReduceProduct, bridge.ReduceProduct, x, axes...)
}

// ===========================================================================
// ArgMin/ArgMax
// ===========================================================================

func (f *Function) ArgMinMax(x backends.Value, axis int, outputDType dtypes.DType, isMin bool) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	// Output shape: same as input but with the reduced axis removed.
	outDims := make([]int, 0, node.shape.Rank()-1)
	for i, d := range node.shape.Dimensions {
		if i != axis {
			outDims = append(outDims, d)
		}
	}
	outShape := shapes.Make(outputDType, outDims...)

	bridgeDType := dtypeToBridgeDType(outputDType)
	var tensor bridge.Tensor
	if isMin {
		tensor, err = f.ctx().ArgMin(node.tensor, axis, bridgeDType)
	} else {
		tensor, err = f.ctx().ArgMax(node.tensor, axis, bridgeDType)
	}
	if err != nil {
		return nil, errors.Wrap(err, "ArgMinMax")
	}
	// MPSGraph keeps reduced dim as size 1 — reshape to squeeze it.
	if outShape.Rank() > 0 {
		squeezeDims := make([]int64, outShape.Rank())
		for i, d := range outShape.Dimensions {
			squeezeDims[i] = int64(d)
		}
		tensor, err = f.ctx().Reshape(tensor, squeezeDims)
		if err != nil {
			return nil, errors.Wrap(err, "ArgMinMax: reshape")
		}
	} else {
		tensor, err = f.ctx().Reshape(tensor, nil)
		if err != nil {
			return nil, errors.Wrap(err, "ArgMinMax: reshape to scalar")
		}
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

// ===========================================================================
// Batch Normalization
// ===========================================================================

func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Value, epsilon float32, featureAxis int) (backends.Value, error) {
	nodes, err := castNodes("BatchNormForInference", operand, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	tensor, err := f.ctx().BatchNormInference(
		nodes[0].tensor, nodes[3].tensor, nodes[4].tensor,
		nodes[1].tensor, nodes[2].tensor,
		float32(epsilon), featureAxis)
	if err != nil {
		return nil, errors.Wrap(err, "BatchNormForInference")
	}
	return &graphNode{tensor: tensor, shape: nodes[0].shape}, nil
}

// ===========================================================================
// Pad
// ===========================================================================

func (f *Function) Pad(operand, fillValue backends.Value, axesConfig ...backends.PadAxis) (backends.Value, error) {
	opNode, err := castNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Pad")
	}
	fillNode, err := castNode(fillValue)
	if err != nil {
		return nil, errors.Wrap(err, "Pad: fillValue")
	}

	// Check for interior padding (not supported by basic MPSGraph pad).
	for _, ac := range axesConfig {
		if ac.Interior != 0 {
			return nil, errors.Errorf("Pad: interior padding not yet supported in MPSGraph backend")
		}
	}

	padBefore := make([]int64, len(axesConfig))
	padAfter := make([]int64, len(axesConfig))
	outDims := make([]int, len(axesConfig))
	for i, ac := range axesConfig {
		padBefore[i] = int64(ac.Start)
		padAfter[i] = int64(ac.End)
		outDims[i] = opNode.shape.Dimensions[i] + ac.Start + ac.End
	}

	tensor, err := f.ctx().Pad(opNode.tensor, fillNode.tensor, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, "Pad")
	}
	outShape := shapes.Make(opNode.shape.DType, outDims...)
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

// ===========================================================================
// DynamicSlice
// ===========================================================================

func (f *Function) DynamicSlice(operand backends.Value, startIndicesValues []backends.Value, sliceSizes []int) (backends.Value, error) {
	opNode, err := castNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice: operand")
	}

	startIndicesTensors := make([]bridge.Tensor, len(startIndicesValues))
	for i, v := range startIndicesValues {
		n, err := castNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "DynamicSlice: startIndex[%d]", i)
		}
		startIndicesTensors[i] = n.tensor
	}

	sliceSizes64 := make([]int64, len(sliceSizes))
	for i, s := range sliceSizes {
		sliceSizes64[i] = int64(s)
	}

	outShape := shapes.Make(opNode.shape.DType, sliceSizes...)
	tensor, err := f.ctx().DynamicSlice(opNode.tensor, startIndicesTensors, sliceSizes64)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicSlice")
	}
	return &graphNode{tensor: tensor, shape: outShape}, nil
}

func (f *Function) DynamicUpdateSlice(operand, update backends.Value, startIndicesValues []backends.Value) (backends.Value, error) {
	opNode, err := castNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: operand")
	}
	updNode, err := castNode(update)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice: update")
	}

	startIndicesTensors := make([]bridge.Tensor, len(startIndicesValues))
	for i, v := range startIndicesValues {
		n, err := castNode(v)
		if err != nil {
			return nil, errors.Wrapf(err, "DynamicUpdateSlice: startIndex[%d]", i)
		}
		startIndicesTensors[i] = n.tensor
	}

	tensor, err := f.ctx().DynamicUpdateSlice(opNode.tensor, updNode.tensor, startIndicesTensors)
	if err != nil {
		return nil, errors.Wrap(err, "DynamicUpdateSlice")
	}
	// Output shape is same as operand shape.
	return &graphNode{tensor: tensor, shape: opNode.shape}, nil
}

// ===========================================================================
// RNG
// ===========================================================================

func (f *Function) RNGBitGenerator(state backends.Value, shape shapes.Shape) (newState, values backends.Value, err error) {
	stateNode, err := castNode(state)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator: state")
	}

	// Validate state shape: expects [3]uint64 per GoMLX convention.
	expectedStateShape := backends.RNGStateShape
	if !stateNode.shape.Equal(expectedStateShape) {
		return nil, nil, errors.Errorf("RNGBitGenerator: expected state shape %s, got %s",
			expectedStateShape, stateNode.shape)
	}

	// Generate random values using MPSGraph's random uniform.
	// MPSGraph manages its own RNG state, so we pass the GoMLX state through unchanged.
	dims := make([]int64, shape.Rank())
	for i, d := range shape.Dimensions {
		dims[i] = int64(d)
	}
	bridgeDType := dtypeToBridgeDType(shape.DType)

	valuesTensor, err := f.ctx().RandomUniform(bridgeDType, dims)
	if err != nil {
		return nil, nil, errors.Wrap(err, "RNGBitGenerator")
	}

	// For integer types, we need random bits, not uniform [0,1).
	// MPSGraph only generates uniform floats, so for integer types we generate
	// Float32 uniform, scale to the range, and cast.
	// However, the common GoMLX pattern is to generate uint32 bits and then convert.
	// For now, we pass the state through and return the random tensor.
	// The state is unchanged since MPSGraph manages its own state.

	newStateNode := &graphNode{tensor: stateNode.tensor, shape: stateNode.shape}
	valuesNode := &graphNode{tensor: valuesTensor, shape: shape}
	return newStateNode, valuesNode, nil
}

// ===========================================================================
// Convolution
// ===========================================================================

func (f *Function) ConvGeneral(
	input, kernel backends.Value,
	axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) (backends.Value, error) {
	inputNode, err := castNode(input)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: input")
	}
	kernelNode, err := castNode(kernel)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: kernel")
	}

	outputShape, err := shapeinference.ConvGeneralOp(
		inputNode.shape, kernelNode.shape, axes, strides, paddings,
		inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	numSpatialDims := len(axes.InputSpatial)
	if numSpatialDims != 2 {
		return nil, errors.Errorf("ConvGeneral: only 2D convolution supported, got %d spatial dims", numSpatialDims)
	}

	// Transpose input and kernel to NCHW / OIHW layout expected by MPSGraph.
	inputTensor, err := f.transposeToNCHW(inputNode, axes.InputBatch, axes.InputChannels, axes.InputSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose input")
	}
	kernelTensor, err := f.transposeToOIHW(kernelNode, axes.KernelOutputChannels, axes.KernelInputChannels, axes.KernelSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose kernel")
	}

	// Prepare strides, dilations, padding.
	strideArr := make([]int64, numSpatialDims)
	dilationArr := make([]int64, numSpatialDims)
	padBeforeArr := make([]int64, numSpatialDims)
	padAfterArr := make([]int64, numSpatialDims)
	for i := range numSpatialDims {
		strideArr[i] = 1
		dilationArr[i] = 1
		if strides != nil && i < len(strides) {
			strideArr[i] = int64(strides[i])
		}
		if kernelDilations != nil && i < len(kernelDilations) {
			dilationArr[i] = int64(kernelDilations[i])
		}
		if paddings != nil && i < len(paddings) {
			padBeforeArr[i] = int64(paddings[i][0])
			padAfterArr[i] = int64(paddings[i][1])
		}
	}

	groups := max(channelGroupCount, 1) * max(batchGroupCount, 1)
	result, err := f.ctx().ConvGeneral(inputTensor, kernelTensor, numSpatialDims,
		strideArr, dilationArr, padBeforeArr, padAfterArr, groups)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	// Transpose output from NCHW back to the requested layout.
	result, err = f.transposeFromNCHW(result, outputShape, axes.OutputBatch, axes.OutputChannels, axes.OutputSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose output")
	}

	return &graphNode{tensor: result, shape: outputShape}, nil
}

// transposeToNCHW transposes a tensor from arbitrary axis layout to NCHW.
func (f *Function) transposeToNCHW(node *graphNode, batchAxis, channelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := node.shape.Rank()
	perm := make([]int, rank)
	perm[0] = batchAxis
	perm[1] = channelAxis
	for i, a := range spatialAxes {
		perm[2+i] = a
	}
	if isIdentityPerm(perm) {
		return node.tensor, nil
	}
	return f.ctx().Transpose(node.tensor, perm)
}

// transposeToOIHW transposes a kernel from arbitrary layout to OIHW.
func (f *Function) transposeToOIHW(node *graphNode, outChannelAxis, inChannelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := node.shape.Rank()
	perm := make([]int, rank)
	perm[0] = outChannelAxis
	perm[1] = inChannelAxis
	for i, a := range spatialAxes {
		perm[2+i] = a
	}
	if isIdentityPerm(perm) {
		return node.tensor, nil
	}
	return f.ctx().Transpose(node.tensor, perm)
}

// transposeFromNCHW transposes from NCHW layout back to the target layout.
func (f *Function) transposeFromNCHW(tensor bridge.Tensor, targetShape shapes.Shape, batchAxis, channelAxis int, spatialAxes []int) (bridge.Tensor, error) {
	rank := targetShape.Rank()
	// Build the inverse permutation: from NCHW position to target position.
	fwdPerm := make([]int, rank)
	fwdPerm[0] = batchAxis
	fwdPerm[1] = channelAxis
	for i, a := range spatialAxes {
		fwdPerm[2+i] = a
	}
	// Compute inverse.
	invPerm := make([]int, rank)
	for i, v := range fwdPerm {
		invPerm[v] = i
	}
	if isIdentityPerm(invPerm) {
		return tensor, nil
	}
	return f.ctx().Transpose(tensor, invPerm)
}

// ===========================================================================
// ReduceWindow (Pooling)
// ===========================================================================

func (f *Function) ReduceWindow(
	x backends.Value,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	outShape, err := shapeinference.ReduceWindowOp(
		node.shape, windowDimensions, strides, baseDilations, windowDilations, paddings)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	// For now, only support the common 2D pooling case with window on spatial dims.
	// Full general ReduceWindow decomposition is complex.
	rank := node.shape.Rank()
	if rank != 4 {
		return nil, errors.Errorf("ReduceWindow: only 4D tensors (NCHW) supported, got rank %d", rank)
	}

	// Check that batch and channel dims have window size 1.
	if windowDimensions[0] != 1 || windowDimensions[1] != 1 {
		return nil, errors.Errorf("ReduceWindow: batch/channel window must be 1, got %v", windowDimensions[:2])
	}

	var mode int
	switch reductionType {
	case backends.ReduceOpMax:
		mode = 0
	case backends.ReduceOpSum:
		mode = 1
	default:
		return nil, errors.Errorf("ReduceWindow: reduction type %v not supported in MPSGraph pooling", reductionType)
	}

	spatialWindow := []int64{int64(windowDimensions[2]), int64(windowDimensions[3])}
	spatialStrides := []int64{1, 1}
	if strides != nil && len(strides) >= 4 {
		spatialStrides = []int64{int64(strides[2]), int64(strides[3])}
	}
	padBefore := []int64{0, 0}
	padAfter := []int64{0, 0}
	if paddings != nil && len(paddings) >= 4 {
		padBefore = []int64{int64(paddings[2][0]), int64(paddings[3][0])}
		padAfter = []int64{int64(paddings[2][1]), int64(paddings[3][1])}
	}

	tensor, err := f.ctx().Pool2D(node.tensor, mode, spatialWindow, spatialStrides, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	// Reshape to match expected output shape if needed.
	outDims := make([]int64, outShape.Rank())
	for i, d := range outShape.Dimensions {
		outDims[i] = int64(d)
	}
	tensor, err = f.ctx().Reshape(tensor, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow: reshape")
	}

	return &graphNode{tensor: tensor, shape: outShape}, nil
}

// ===========================================================================
// TotalOrder Comparisons
// ===========================================================================
// TotalOrder comparisons enforce: -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN.
// For simplicity, we delegate to regular comparisons (correct for non-NaN values,
// which is the common case in ML).

func (f *Function) EqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.Equal(lhs, rhs)
}

func (f *Function) NotEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.NotEqual(lhs, rhs)
}

func (f *Function) GreaterThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.GreaterThan(lhs, rhs)
}

func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.GreaterOrEqual(lhs, rhs)
}

func (f *Function) LessThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.LessThan(lhs, rhs)
}

func (f *Function) LessOrEqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error) {
	return f.LessOrEqual(lhs, rhs)
}

// ===========================================================================
// Logical / Bitwise Reductions
// ===========================================================================

func (f *Function) ReduceLogicalAnd(x backends.Value, axes ...int) (backends.Value, error) {
	// ReduceMin of {0,1} gives AND semantics for boolean values.
	return f.reduceOp("ReduceLogicalAnd", backends.OpTypeReduceLogicalAnd, bridge.ReduceMin, x, axes...)
}

func (f *Function) ReduceLogicalOr(x backends.Value, axes ...int) (backends.Value, error) {
	// ReduceMax of {0,1} gives OR semantics for boolean values.
	return f.reduceOp("ReduceLogicalOr", backends.OpTypeReduceLogicalOr, bridge.ReduceMax, x, axes...)
}

// ===========================================================================
// Fused Operations
// ===========================================================================

func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	tensor, err := f.ctx().Softmax(node.tensor, axis)
	if err != nil {
		return nil, errors.Wrap(err, "FusedSoftmax")
	}
	return &graphNode{tensor: tensor, shape: node.shape}, nil
}

func (f *Function) FusedGelu(x backends.Value, exact bool) (backends.Value, error) {
	return nil, errors.Wrap(backends.ErrNotImplemented, "FusedGelu")
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	return nil, errors.Wrap(backends.ErrNotImplemented, "FusedLayerNorm")
}

func (f *Function) FusedDense(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.Wrap(backends.ErrNotImplemented, "FusedDense")
}

func (f *Function) FusedScaledDotProductAttention(
	query, key, value, mask backends.Value,
	numHeads, numKVHeads int,
	axesLayout backends.AxesLayout,
	scale float64,
	causal bool,
) (backends.Value, error) {
	return nil, errors.Wrap(backends.ErrNotImplemented, "FusedScaledDotProductAttention")
}

func (f *Function) FusedAttentionQKVProjection(
	x, wQKV, biasQ, biasK, biasV backends.Value,
	queryDim, keyValueDim int,
) (query, key, value backends.Value, err error) {
	return nil, nil, nil, errors.Wrap(backends.ErrNotImplemented, "FusedAttentionQKVProjection")
}
