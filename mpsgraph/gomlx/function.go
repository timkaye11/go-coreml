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
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	tensor, err := f.ctx().Identity(node.tensor)
	if err != nil {
		return nil, errors.Wrap(err, "Identity")
	}
	return &graphNode{tensor: tensor, shape: node.shape}, nil
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
	// Empty axes means reduce all dimensions.
	if len(axes) == 0 {
		axes = make([]int, node.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
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

	dims := make([]int64, shape.Rank())
	for i, d := range shape.Dimensions {
		dims[i] = int64(d)
	}

	// GoMLX's RandomUniform calls RNGBitGenerator with Uint32 dtype to get random bits,
	// then converts to float: ConvertDType(bits, Float32) * (1/2^32).
	// MPSGraph only supports float types for random generation (float16, bfloat16, float32).
	// Strategy: generate Float32 uniform [0, 1), scale to [0, 2^32) so the subsequent
	// ConvertDType(Uint32→Float32) is a no-op cast and MulScalar(1/2^32) produces [0, 1).
	var valuesTensor bridge.Tensor
	if shape.DType.IsFloat() {
		// Direct generation for float types.
		bridgeDType := dtypeToBridgeDType(shape.DType)
		valuesTensor, err = f.ctx().RandomUniform(bridgeDType, dims)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator")
		}
	} else {
		// Integer type (typically Uint32): generate Float32 uniform and scale.
		valuesTensor, err = f.ctx().RandomUniform(dtypeToBridgeDType(dtypes.Float32), dims)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: RandomUniform(Float32)")
		}
		// Scale [0, 1) → [0, 2^32) so the calling code's pipeline
		// (ConvertDType + MulScalar(1/2^32)) produces correct [0, 1) uniform.
		scaleVal := float32(4294967296.0) // 2^32
		scaleTensor, err := f.ctx().Constant(
			unsafe.Pointer(&scaleVal), 4, dtypeToBridgeDType(dtypes.Float32), []int64{1})
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale constant")
		}
		valuesTensor, err = f.ctx().Mul(valuesTensor, scaleTensor)
		if err != nil {
			return nil, nil, errors.Wrap(err, "RNGBitGenerator: scale")
		}
	}

	// Pass the GoMLX RNG state through unchanged (MPSGraph manages its own state).
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

	numSpatialDims := len(axes.InputSpatial)

	// Default nil strides to 1.
	if strides == nil {
		strides = make([]int, numSpatialDims)
		for i := range strides {
			strides[i] = 1
		}
	}

	// Default nil dilations to 1.
	if inputDilations == nil {
		inputDilations = make([]int, numSpatialDims)
		for i := range inputDilations {
			inputDilations[i] = 1
		}
	}
	if kernelDilations == nil {
		kernelDilations = make([]int, numSpatialDims)
		for i := range kernelDilations {
			kernelDilations[i] = 1
		}
	}

	// Default group counts.
	if channelGroupCount < 1 {
		channelGroupCount = 1
	}
	if batchGroupCount < 1 {
		batchGroupCount = 1
	}

	outputShape, err := shapeinference.ConvGeneralOp(
		inputNode.shape, kernelNode.shape, axes, strides, paddings,
		inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral")
	}

	if numSpatialDims != 2 {
		return nil, errors.Errorf("ConvGeneral: only 2D convolution supported, got %d spatial dims", numSpatialDims)
	}

	// Check if input dilation is needed (values > 1).
	hasInputDilation := false
	for _, d := range inputDilations {
		if d > 1 {
			hasInputDilation = true
			break
		}
	}

	// Transpose input and kernel to NCHW / OIHW layout expected by MPSGraph.
	inputTensor, err := f.transposeToNCHW(inputNode, axes.InputBatch, axes.InputChannels, axes.InputSpatial)
	if err != nil {
		return nil, errors.Wrap(err, "ConvGeneral: transpose input")
	}

	// Handle input dilation by inserting zeros between input elements.
	// Input dilation of D for an axis means: between each pair of values, insert (D-1) zeros.
	// This expands a dimension of size N to (N-1)*D + 1.
	if hasInputDilation {
		inputTensor, err = f.dilateInput(inputTensor, inputNode.shape, axes.InputBatch, axes.InputChannels, axes.InputSpatial, inputDilations)
		if err != nil {
			return nil, errors.Wrap(err, "ConvGeneral: input dilation")
		}
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
		strideArr[i] = int64(strides[i])
		dilationArr[i] = int64(kernelDilations[i])
		if paddings != nil && i < len(paddings) {
			padBeforeArr[i] = int64(paddings[i][0])
			padAfterArr[i] = int64(paddings[i][1])
		}
	}

	groups := channelGroupCount * batchGroupCount
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

// dilateInput inserts zeros between input elements for input dilation.
// Input is already in NCHW layout. dilations are per spatial axis.
func (f *Function) dilateInput(tensor bridge.Tensor, origShape shapes.Shape, batchAxis, channelAxis int, spatialAxes []int, dilations []int) (bridge.Tensor, error) {
	// After transpose to NCHW, spatial dims are at indices 2 and 3.
	// Dilation of D on an axis with size N → new size = (N-1)*D + 1.
	// We use Pad with interior padding to achieve this.
	origDims := origShape.Dimensions
	// Get spatial dims in original order.
	spatialSizes := make([]int64, len(spatialAxes))
	for i, ax := range spatialAxes {
		spatialSizes[i] = int64(origDims[ax])
	}

	// Compute dilated sizes and pad amounts.
	// In NCHW layout: [batch, channels, H, W], spatial at indices 2, 3.
	// Build padBefore/padAfter arrays with interior padding.
	// MPSGraph pad doesn't support interior padding, so we build with Iota + scatter approach.
	// Actually, a simpler approach: create a zero tensor of the dilated size and scatter original values.

	batchSize := int64(origDims[batchAxis])
	channelSize := int64(origDims[channelAxis])
	dilatedH := (spatialSizes[0]-1)*int64(dilations[0]) + 1
	dilatedW := (spatialSizes[1]-1)*int64(dilations[1]) + 1

	// Create a zero tensor of the dilated size [batch, channels, dilatedH, dilatedW].
	zeroVal := float32(0)
	zeroTensor, err := f.ctx().Constant(
		unsafe.Pointer(&zeroVal), 4, dtypeToBridgeDType(origShape.DType), []int64{1})
	if err != nil {
		return nil, errors.Wrap(err, "dilateInput: zero constant")
	}
	dilatedShape := []int64{batchSize, channelSize, dilatedH, dilatedW}
	zeroTensor, err = f.ctx().BroadcastTo(zeroTensor, dilatedShape)
	if err != nil {
		return nil, errors.Wrap(err, "dilateInput: broadcast zeros")
	}

	// Use slice + dynamic_update_slice to place original values at strided positions.
	// Actually, the simplest approach: use Pad with 0 before, 0 after, and (dilation-1) interior.
	// But our bridge doesn't support interior padding.

	// Alternative: create with strides using Slice in reverse.
	// Actually the simplest correct approach for input dilation:
	// Build indices for scattered positions and use gather/scatter.
	// But that's complex. Let me use a different approach:
	// Reshape + interleave with zeros using Concatenate along spatial axes.

	// Simplest approach: iterate and build with concat.
	// For moderate dilation factors, this is reasonable.

	// Actually, let me just implement this with a strided assignment pattern using
	// DynamicUpdateSlice. For each row/col, update the appropriate position.

	// The most efficient approach: use pad with interior padding.
	// We can implement interior padding as: create dilated zero tensor, then
	// for each (h, w) in original, place at (h*dilH, w*dilW) in dilated.
	// This is a gather operation: create stride indices.

	// Actually, the cleanest approach: use Slice with negative strides (not supported),
	// or simply use a workaround.

	// Let me try: create the dilated tensor directly using the stridedSlice approach:
	// tensor is [B, C, H, W], we want [B, C, (H-1)*d+1, (W-1)*d+1]
	// with original values at positions [0, d, 2d, ...] in each spatial axis.

	// The simplest correct approach uses Iota to generate scatter indices:
	// For now, just create a Pad operation that inserts zeros.
	// Our Pad bridge doesn't support interior padding, so let's implement it
	// by reshaping + concat.

	// For dilation D on axis of size N:
	//   1. Reshape: [..., N, 1, ...]
	//   2. Pad with D-1 zeros on the last new dim: [..., N, D, ...]
	//   3. Reshape to flatten: [..., N*D, ...]
	//   4. Slice to remove trailing D-1 zeros: [..., (N-1)*D+1, ...]

	result := tensor

	// Dilate height (axis 2 in NCHW).
	if dilations[0] > 1 {
		result, err = f.dilateAxis(result, 2, spatialSizes[0], int64(dilations[0]),
			[]int64{batchSize, channelSize, spatialSizes[0], spatialSizes[1]}, origShape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "dilateInput: dilate H")
		}
		spatialSizes[0] = dilatedH
	}

	// Dilate width (axis 3 in NCHW).
	if dilations[1] > 1 {
		result, err = f.dilateAxis(result, 3, spatialSizes[1], int64(dilations[1]),
			[]int64{batchSize, channelSize, dilatedH, spatialSizes[1]}, origShape.DType)
		if err != nil {
			return nil, errors.Wrap(err, "dilateInput: dilate W")
		}
	}

	return result, nil
}

// dilateAxis dilates a single axis by inserting (dilation-1) zeros between elements.
// Approach: reshape to insert a new dim, pad that dim, reshape to flatten, then slice.
func (f *Function) dilateAxis(tensor bridge.Tensor, axis int, axisSize, dilation int64, currentShape []int64, dtype dtypes.DType) (bridge.Tensor, error) {
	rank := len(currentShape)

	// Step 1: Reshape to split the target axis into [axisSize, 1].
	reshapeDims := make([]int64, rank+1)
	for i := 0; i < axis; i++ {
		reshapeDims[i] = currentShape[i]
	}
	reshapeDims[axis] = axisSize
	reshapeDims[axis+1] = 1
	for i := axis + 1; i < rank; i++ {
		reshapeDims[i+1] = currentShape[i]
	}
	result, err := f.ctx().Reshape(tensor, reshapeDims)
	if err != nil {
		return nil, err
	}

	// Step 2: Pad the new axis (axis+1) with (dilation-1) zeros after.
	padBefore := make([]int64, rank+1)
	padAfter := make([]int64, rank+1)
	padAfter[axis+1] = dilation - 1

	zeroVal := float32(0)
	zeroTensor, err := f.ctx().Constant(
		unsafe.Pointer(&zeroVal), 4, dtypeToBridgeDType(dtype), []int64{1})
	if err != nil {
		return nil, err
	}
	// Reshape zero to scalar for pad.
	zeroTensor, err = f.ctx().Reshape(zeroTensor, nil)
	if err != nil {
		return nil, err
	}

	result, err = f.ctx().Pad(result, zeroTensor, padBefore, padAfter)
	if err != nil {
		return nil, err
	}

	// Step 3: Reshape to flatten the axis back: [axisSize * dilation].
	flatDims := make([]int64, rank)
	for i := 0; i < axis; i++ {
		flatDims[i] = currentShape[i]
	}
	flatDims[axis] = axisSize * dilation
	for i := axis + 1; i < rank; i++ {
		flatDims[i] = currentShape[i]
	}
	result, err = f.ctx().Reshape(result, flatDims)
	if err != nil {
		return nil, err
	}

	// Step 4: Slice to remove trailing (dilation-1) zeros.
	// New size = (axisSize-1)*dilation + 1.
	dilatedSize := (axisSize-1)*dilation + 1
	starts := make([]int64, rank)
	ends := make([]int64, rank)
	strides := make([]int64, rank)
	for i := range rank {
		ends[i] = flatDims[i]
		strides[i] = 1
	}
	ends[axis] = dilatedSize
	result, err = f.ctx().Slice(result, starts, ends, strides)
	if err != nil {
		return nil, err
	}

	return result, nil
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

// poolAxesInfo holds axis mapping for pool operations.
// MPSGraph pool2d expects NCHW layout; this detects the actual layout
// and provides permutations for transposing to/from NCHW.
type poolAxesInfo struct {
	spatialAxes    [2]int   // Indices of spatial axes in original layout.
	nonSpatialAxes [2]int   // Indices of batch/channel axes in original layout.
	toNCHW         []int    // Permutation from original layout to NCHW.
	fromNCHW       []int    // Permutation from NCHW back to original layout.
	needsTranspose bool     // Whether transposition is needed.
	spatialWindow  [2]int64 // Window sizes for spatial dims.
	spatialStrides [2]int64 // Strides for spatial dims.
	padBefore      [2]int64 // Padding before for spatial dims.
	padAfter       [2]int64 // Padding after for spatial dims.
}

// detectPoolAxes detects spatial axes from windowDimensions and builds
// transposition info. Spatial axes are those with window > 1 or stride > 1
// or non-zero padding.
func detectPoolAxes(windowDimensions, windowStrides []int, paddings [][2]int) (poolAxesInfo, error) {
	var info poolAxesInfo

	// Detect spatial axes: those with window > 1 or stride > 1 or padding.
	var spatialAxes, nonSpatialAxes []int
	for i := range 4 {
		isSpatial := false
		if windowDimensions[i] > 1 {
			isSpatial = true
		}
		if windowStrides != nil && i < len(windowStrides) && windowStrides[i] > 1 {
			isSpatial = true
		}
		if paddings != nil && i < len(paddings) && (paddings[i][0] != 0 || paddings[i][1] != 0) {
			isSpatial = true
		}
		if isSpatial {
			spatialAxes = append(spatialAxes, i)
		} else {
			nonSpatialAxes = append(nonSpatialAxes, i)
		}
	}

	if len(spatialAxes) != 2 || len(nonSpatialAxes) != 2 {
		return info, errors.Errorf("expected exactly 2 spatial axes (window > 1), got %d spatial %v, %d non-spatial %v",
			len(spatialAxes), spatialAxes, len(nonSpatialAxes), nonSpatialAxes)
	}

	info.spatialAxes = [2]int{spatialAxes[0], spatialAxes[1]}
	info.nonSpatialAxes = [2]int{nonSpatialAxes[0], nonSpatialAxes[1]}

	// Build permutation to NCHW: [nonSpatial0, nonSpatial1, spatial0, spatial1].
	info.toNCHW = []int{nonSpatialAxes[0], nonSpatialAxes[1], spatialAxes[0], spatialAxes[1]}
	info.needsTranspose = info.toNCHW[0] != 0 || info.toNCHW[1] != 1 || info.toNCHW[2] != 2 || info.toNCHW[3] != 3

	// Inverse permutation.
	info.fromNCHW = make([]int, 4)
	for i, v := range info.toNCHW {
		info.fromNCHW[v] = i
	}

	// Extract spatial parameters.
	info.spatialWindow = [2]int64{int64(windowDimensions[spatialAxes[0]]), int64(windowDimensions[spatialAxes[1]])}
	info.spatialStrides = [2]int64{1, 1}
	if windowStrides != nil {
		info.spatialStrides = [2]int64{int64(windowStrides[spatialAxes[0]]), int64(windowStrides[spatialAxes[1]])}
	}
	if paddings != nil {
		info.padBefore = [2]int64{int64(paddings[spatialAxes[0]][0]), int64(paddings[spatialAxes[1]][0])}
		info.padAfter = [2]int64{int64(paddings[spatialAxes[0]][1]), int64(paddings[spatialAxes[1]][1])}
	}

	return info, nil
}

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

	rank := node.shape.Rank()
	if rank != 4 {
		return nil, errors.Errorf("ReduceWindow: only 4D tensors supported, got rank %d", rank)
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

	axesInfo, err := detectPoolAxes(windowDimensions, strides, paddings)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	// Transpose to NCHW if needed.
	tensor := node.tensor
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: transpose to NCHW")
		}
	}

	spatialWindow := axesInfo.spatialWindow[:]
	spatialStrides := axesInfo.spatialStrides[:]
	padBefore := axesInfo.padBefore[:]
	padAfter := axesInfo.padAfter[:]

	tensor, err = f.ctx().Pool2D(tensor, mode, spatialWindow, spatialStrides, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, "ReduceWindow")
	}

	// Transpose back from NCHW if needed.
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.fromNCHW)
		if err != nil {
			return nil, errors.Wrap(err, "ReduceWindow: transpose from NCHW")
		}
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
// SelectAndScatter (MaxPool gradient)
// ===========================================================================

func (f *Function) SelectAndScatterMax(operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	return f.selectAndScatterImpl("SelectAndScatterMax", operand, source, windowDimensions, windowStrides, paddings)
}

func (f *Function) SelectAndScatterMin(operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	return nil, errors.Errorf("SelectAndScatterMin not yet supported in MPSGraph backend")
}

func (f *Function) selectAndScatterImpl(opName string, operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	opNode, err := castNode(operand)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: operand", opName)
	}
	srcNode, err := castNode(source)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: source", opName)
	}

	rank := opNode.shape.Rank()
	if rank != 4 {
		return nil, errors.Errorf("%s: only 4D tensors supported, got rank %d", opName, rank)
	}

	axesInfo, err := detectPoolAxes(windowDimensions, windowStrides, paddings)
	if err != nil {
		return nil, errors.Wrapf(err, "%s", opName)
	}

	// Transpose operand and source to NCHW if needed.
	opTensor := opNode.tensor
	srcTensor := srcNode.tensor
	if axesInfo.needsTranspose {
		opTensor, err = f.ctx().Transpose(opTensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose operand to NCHW", opName)
		}
		srcTensor, err = f.ctx().Transpose(srcTensor, axesInfo.toNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose source to NCHW", opName)
		}
	}

	spatialWindow := axesInfo.spatialWindow[:]
	spatialStrides := axesInfo.spatialStrides[:]
	padBefore := axesInfo.padBefore[:]
	padAfter := axesInfo.padAfter[:]

	// MPSGraph's maxPooling2DGradient takes:
	// - gradient: the incoming gradient (same shape as pool output = source)
	// - sourceTensor: the original input to pooling (= operand)
	tensor, err := f.ctx().MaxPool2DGradient(srcTensor, opTensor, spatialWindow, spatialStrides, padBefore, padAfter)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	// Transpose back from NCHW if needed.
	if axesInfo.needsTranspose {
		tensor, err = f.ctx().Transpose(tensor, axesInfo.fromNCHW)
		if err != nil {
			return nil, errors.Wrapf(err, "%s: transpose from NCHW", opName)
		}
	}

	// Output shape is same as operand.
	return &graphNode{tensor: tensor, shape: opNode.shape}, nil
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
	// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	dt := node.shape.DType

	makeConst := func(val float32) (bridge.Tensor, error) {
		t, err := f.ctx().Constant(unsafe.Pointer(&val), 4, dtypeToBridgeDType(dt), []int64{1})
		if err != nil {
			return nil, err
		}
		return f.ctx().Reshape(t, nil) // scalar
	}

	half, err := makeConst(0.5)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	one, err := makeConst(1.0)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	coeff, err := makeConst(0.044715)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	sqrtTwoPi, err := makeConst(0.7978845608) // sqrt(2/pi)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}

	t := node.tensor
	// x^3
	x2, err := f.ctx().Mul(t, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	x3, err := f.ctx().Mul(x2, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.044715 * x^3
	cx3, err := f.ctx().Mul(coeff, x3)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// x + 0.044715 * x^3
	inner, err := f.ctx().Add(t, cx3)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled, err := f.ctx().Mul(sqrtTwoPi, inner)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// tanh(...)
	tanhVal, err := f.ctx().Tanh(scaled)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 1 + tanh(...)
	onePlusTanh, err := f.ctx().Add(one, tanhVal)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.5 * x
	halfX, err := f.ctx().Mul(half, t)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}
	// 0.5 * x * (1 + tanh(...))
	result, err := f.ctx().Mul(halfX, onePlusTanh)
	if err != nil {
		return nil, errors.Wrap(err, "FusedGelu")
	}

	return &graphNode{tensor: result, shape: node.shape}, nil
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	node, err := castNode(x)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm")
	}

	// Normalize negative axes.
	rank := node.shape.Rank()
	normalizedAxes := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, errors.Errorf("FusedLayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normalizedAxes[i] = ax
	}

	// Compute mean over the specified axes.
	meanTensor, err := f.ctx().Reduce(node.tensor, bridge.ReduceSum, normalizedAxes)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: reduce for mean")
	}

	// Count elements being reduced.
	numElements := int64(1)
	for _, ax := range normalizedAxes {
		numElements *= int64(node.shape.Dimensions[ax])
	}
	dt := node.shape.DType
	countVal := float32(numElements)
	countTensor, err := f.ctx().Constant(unsafe.Pointer(&countVal), 4, dtypeToBridgeDType(dt), []int64{1})
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: count constant")
	}
	countTensor, err = f.ctx().Reshape(countTensor, nil) // scalar
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: count reshape")
	}

	// mean = sum / count
	meanTensor, err = f.ctx().Div(meanTensor, countTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: mean div")
	}

	// x - mean (broadcast automatically)
	diff, err := f.ctx().Sub(node.tensor, meanTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: subtract mean")
	}

	// variance = mean((x - mean)^2)
	diffSq, err := f.ctx().Mul(diff, diff)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: diff squared")
	}
	varTensor, err := f.ctx().Reduce(diffSq, bridge.ReduceSum, normalizedAxes)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: reduce for variance")
	}
	varTensor, err = f.ctx().Div(varTensor, countTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: variance div")
	}

	// variance + epsilon
	epsVal := float32(epsilon)
	epsTensor, err := f.ctx().Constant(unsafe.Pointer(&epsVal), 4, dtypeToBridgeDType(dt), []int64{1})
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: epsilon constant")
	}
	epsTensor, err = f.ctx().Reshape(epsTensor, nil) // scalar
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: epsilon reshape")
	}
	varPlusEps, err := f.ctx().Add(varTensor, epsTensor)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: var + eps")
	}

	// 1 / sqrt(variance + epsilon)
	invStd, err := f.ctx().Rsqrt(varPlusEps)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: rsqrt")
	}

	// normalized = (x - mean) * invStd
	normalized, err := f.ctx().Mul(diff, invStd)
	if err != nil {
		return nil, errors.Wrap(err, "FusedLayerNorm: normalize")
	}

	// Build int64 shape for broadcasting.
	targetShape := make([]int64, rank)
	for i, d := range node.shape.Dimensions {
		targetShape[i] = int64(d)
	}

	// broadcastToTarget reshapes a lower-rank tensor (e.g. gamma [4]) to the
	// target shape by inserting size-1 dims for non-normalized axes, then broadcasting.
	broadcastToTarget := func(t bridge.Tensor, tShape shapes.Shape) (bridge.Tensor, error) {
		if tShape.Rank() >= rank {
			return t, nil
		}
		// Build reshape: insert 1s for non-normalized axes.
		reshapeDims := make([]int64, rank)
		normIdx := 0
		for i := range rank {
			isNormAxis := false
			for _, ax := range normalizedAxes {
				if ax == i {
					isNormAxis = true
					break
				}
			}
			if isNormAxis && normIdx < tShape.Rank() {
				reshapeDims[i] = int64(tShape.Dimensions[normIdx])
				normIdx++
			} else {
				reshapeDims[i] = 1
			}
		}
		reshaped, err := f.ctx().Reshape(t, reshapeDims)
		if err != nil {
			return nil, err
		}
		broadcasted, err := f.ctx().BroadcastTo(reshaped, targetShape)
		if err != nil {
			return nil, err
		}
		return broadcasted, nil
	}

	// Apply gamma (scale) if provided.
	if gamma != nil {
		gammaNode, err := castNode(gamma)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: gamma")
		}
		gammaTensor, err := broadcastToTarget(gammaNode.tensor, gammaNode.shape)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: broadcast gamma")
		}
		normalized, err = f.ctx().Mul(normalized, gammaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: apply gamma")
		}
	}

	// Apply beta (offset) if provided.
	if beta != nil {
		betaNode, err := castNode(beta)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: beta")
		}
		betaTensor, err := broadcastToTarget(betaNode.tensor, betaNode.shape)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: broadcast beta")
		}
		normalized, err = f.ctx().Add(normalized, betaTensor)
		if err != nil {
			return nil, errors.Wrap(err, "FusedLayerNorm: apply beta")
		}
	}

	return &graphNode{tensor: normalized, shape: node.shape}, nil
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
