// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"sync"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// AutogradCompileOptions configures MLX-native autograd compilation.
//
// This is intentionally narrower than the normal Compile path:
// control flow and Go-callback graphs are rejected for now because they do not
// yet have a robust large-training autograd story in this backend.
type AutogradCompileOptions struct {
	// Checkpoint wraps the forward closure with MLX checkpointing so backward
	// recomputes intermediates instead of retaining them from the forward pass.
	Checkpoint bool
}

// AutogradBuilder is an MLX-specific extension that exposes MLX-native
// differentiation for training-oriented code.
type AutogradBuilder interface {
	backends.Builder

	// CompileValueAndGrad compiles the builder's main function into an MLX-native
	// value-and-grad executable. The main function must return exactly one scalar
	// output, which is interpreted as the training loss.
	//
	// If argnums is empty, gradients are computed for all inputs.
	CompileValueAndGrad(argnums []int, opts AutogradCompileOptions) (ValueAndGradExecutable, error)

	// CompileCheckpointedValueAndGrad is a convenience wrapper around
	// CompileValueAndGrad(..., AutogradCompileOptions{Checkpoint: true}).
	CompileCheckpointedValueAndGrad(argnums []int) (ValueAndGradExecutable, error)
}

// ValueAndGradExecutable is the MLX-native training-oriented counterpart to a
// normal forward executable. It returns both the scalar loss value(s) and the
// gradients with respect to the selected input arguments.
type ValueAndGradExecutable interface {
	Finalize()
	Inputs() (names []string, inputShapes []shapes.Shape)
	Outputs() (outputShapes []shapes.Shape)
	GradientInputs() []int
	GradientShapes() []shapes.Shape
	UsesCheckpoint() bool
	ExecuteValueAndGrad(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) (values []backends.Buffer, grads []backends.Buffer, err error)
}

var _ AutogradBuilder = (*Builder)(nil)

type valueAndGradExecutable struct {
	backend           *Backend
	mainFn            *Function
	baseClosure       *bridge.Closure
	checkpointClosure *bridge.Closure
	valueAndGrad      *bridge.ValueAndGradClosure
	inputNames        []string
	inputShapes       []shapes.Shape
	outputShapes      []shapes.Shape
	gradArgnums       []int
	gradShapes        []shapes.Shape
	mode              ExecutionMode
	reasons           []string
	checkpointed      bool
	mu                sync.Mutex
}

var _ ValueAndGradExecutable = (*valueAndGradExecutable)(nil)
var _ ExecutionModeReporter = (*valueAndGradExecutable)(nil)

func (b *Builder) CompileCheckpointedValueAndGrad(argnums []int) (ValueAndGradExecutable, error) {
	return b.CompileValueAndGrad(argnums, AutogradCompileOptions{Checkpoint: true})
}

func (b *Builder) CompileValueAndGrad(argnums []int, opts AutogradCompileOptions) (ValueAndGradExecutable, error) {
	if b.compiled {
		return nil, errors.New("Builder already compiled")
	}
	if b.mainFn == nil {
		return nil, errors.New("Builder has no main function")
	}
	if !b.mainFn.returned {
		return nil, errors.New("Main function has no Return() call")
	}
	b.compiled = true

	spec, err := b.compileAutogradClosure()
	if err != nil {
		return nil, err
	}

	if len(spec.outputShapes) != 1 || !spec.outputShapes[0].IsScalar() {
		spec.baseClosure.Free()
		b.mainFn.finalize()
		return nil, errors.Errorf(
			"mlx: CompileValueAndGrad requires the main function to return exactly one scalar loss, got %v",
			spec.outputShapes,
		)
	}

	normalizedArgnums, err := normalizeArgnums(argnums, len(spec.inputShapes))
	if err != nil {
		spec.baseClosure.Free()
		b.mainFn.finalize()
		return nil, err
	}

	autogradClosure := spec.baseClosure
	var checkpointClosure *bridge.Closure
	if opts.Checkpoint {
		checkpointClosure = bridge.Checkpoint(spec.baseClosure)
		autogradClosure = checkpointClosure
	}
	vg := bridge.ValueAndGrad(autogradClosure, normalizedArgnums)

	gradShapes := make([]shapes.Shape, len(normalizedArgnums))
	for ii, argnum := range normalizedArgnums {
		gradShapes[ii] = spec.inputShapes[argnum]
	}

	if b.backend.config.LogExecution {
		reasons := append([]string(nil), spec.reasons...)
		reasons = append(reasons, "value_and_grad")
		if opts.Checkpoint {
			reasons = append(reasons, "checkpoint")
		}
		b.backend.logExecutionMode(b.name, spec.mode, reasons)
	}

	return &valueAndGradExecutable{
		backend:           b.backend,
		mainFn:            b.mainFn,
		baseClosure:       spec.baseClosure,
		checkpointClosure: checkpointClosure,
		valueAndGrad:      vg,
		inputNames:        spec.inputNames,
		inputShapes:       spec.inputShapes,
		outputShapes:      spec.outputShapes,
		gradArgnums:       normalizedArgnums,
		gradShapes:        gradShapes,
		mode:              spec.mode,
		reasons:           spec.reasons,
		checkpointed:      opts.Checkpoint,
	}, nil
}

type autogradCompileSpec struct {
	baseClosure  *bridge.Closure
	inputNames   []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
	mode         ExecutionMode
	reasons      []string
}

func (b *Builder) compileAutogradClosure() (*autogradCompileSpec, error) {
	if b.mainFn.controlFlowStep != nil {
		return nil, errors.Errorf("mlx: MLX-native autograd does not yet support control flow in builder %q", b.name)
	}

	inputNames, inputShapes := collectParamInfo(b.mainFn.params)
	mainFn := b.mainFn
	backend := b.backend
	reasons := mainFn.goCallbackReasons()

	if mainFn.hasGoCallback {
		mode := ExecutionModeRawGoClosure
		if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
			return nil, err
		}
		return nil, errors.Errorf(
			"mlx: MLX-native autograd does not support Go-callback graphs in builder %q (%v)",
			b.name, reasons,
		)
	}

	// If the graph is fully serializable to the C tape interpreter, prefer that.
	// It avoids Go callback overhead and gives MLX a backend-native closure.
	if len(mainFn.instrs) > 0 {
		mode := ExecutionModeCTapeInterpreter
		if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
			return nil, err
		}
		instrs, numSlots, outputSlots, constSlots, constArrays, paramSlots := mainFn.serializeTape()
		return &autogradCompileSpec{
			baseClosure:  bridge.NewClosureFromCTape(instrs, numSlots, outputSlots, constSlots, constArrays, paramSlots),
			inputNames:   inputNames,
			inputShapes:  inputShapes,
			outputShapes: collectOutputShapes(mainFn.outputs),
			mode:         mode,
			reasons:      reasons,
		}, nil
	}

	// For non-C-tape graphs that still avoid Go-callback ops, use a replay closure
	// that frees intermediates on every call. Unlike the normal forward compile path,
	// this closure is executed at runtime by MLX autograd rather than only once for
	// tracing, so it must not retain intermediates across invocations.
	replayFn := func(inputs []*bridge.Array) []*bridge.Array {
		s := backend.stream()
		arrays := replayTape(mainFn, inputs, s)

		outputs := make([]*bridge.Array, len(mainFn.outputs))
		outputIndices := make(map[int]bool, len(mainFn.outputs))
		for ii, out := range mainFn.outputs {
			outputs[ii] = arrays[out.tapeIdx]
			outputIndices[out.tapeIdx] = true
		}
		freeIntermediates(mainFn, arrays, outputIndices)
		return outputs
	}

	mode := ExecutionModeCompiledGoClosure
	if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
		return nil, err
	}
	return &autogradCompileSpec{
		baseClosure:  bridge.NewClosureFromGoFunc(replayFn),
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: collectOutputShapes(mainFn.outputs),
		mode:         mode,
		reasons:      reasons,
	}, nil
}

func normalizeArgnums(argnums []int, numInputs int) ([]int, error) {
	if numInputs == 0 {
		return nil, errors.New("mlx: CompileValueAndGrad requires at least one input")
	}

	if len(argnums) == 0 {
		argnums = make([]int, numInputs)
		for ii := range argnums {
			argnums[ii] = ii
		}
	}

	seen := make(map[int]struct{}, len(argnums))
	normalized := make([]int, 0, len(argnums))
	for _, argnum := range argnums {
		if argnum < 0 || argnum >= numInputs {
			return nil, errors.Errorf("mlx: invalid argnum %d for %d inputs", argnum, numInputs)
		}
		if _, found := seen[argnum]; found {
			return nil, errors.Errorf("mlx: duplicate argnum %d", argnum)
		}
		seen[argnum] = struct{}{}
		normalized = append(normalized, argnum)
	}
	return normalized, nil
}

func (e *valueAndGradExecutable) Finalize() {
	if e.valueAndGrad != nil {
		e.valueAndGrad.Free()
		e.valueAndGrad = nil
	}
	if e.checkpointClosure != nil {
		e.checkpointClosure.Free()
		e.checkpointClosure = nil
	}
	if e.baseClosure != nil {
		e.baseClosure.Free()
		e.baseClosure = nil
	}
	if e.mainFn != nil {
		e.mainFn.finalize()
	}
}

func (e *valueAndGradExecutable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *valueAndGradExecutable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

func (e *valueAndGradExecutable) GradientInputs() []int {
	return append([]int(nil), e.gradArgnums...)
}

func (e *valueAndGradExecutable) GradientShapes() []shapes.Shape {
	return append([]shapes.Shape(nil), e.gradShapes...)
}

func (e *valueAndGradExecutable) ExecutionMode() ExecutionMode {
	return e.mode
}

func (e *valueAndGradExecutable) ExecutionReasons() []string {
	if len(e.reasons) == 0 {
		return nil
	}
	return append([]string(nil), e.reasons...)
}

func (e *valueAndGradExecutable) UsesCheckpoint() bool {
	return e.checkpointed
}

func (e *valueAndGradExecutable) ExecuteValueAndGrad(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) (values []backends.Buffer, grads []backends.Buffer, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, nil, errors.Errorf("ExecuteValueAndGrad: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	paramArrays := make([]*bridge.Array, len(inputs))
	for ii, input := range inputs {
		buf, ok := input.(*mlxBuffer)
		if !ok {
			return nil, nil, errors.Errorf("ExecuteValueAndGrad: input #%d is not a *mlxBuffer, got %T", ii, input)
		}
		if !buf.shape.Equal(e.inputShapes[ii]) {
			return nil, nil, errors.Errorf(
				"ExecuteValueAndGrad: input #%d shape mismatch: expected %s, got %s",
				ii, e.inputShapes[ii], buf.shape,
			)
		}
		paramArrays[ii] = buf.array
	}

	valueArrays, gradArrays, err := e.valueAndGrad.Apply(paramArrays)
	if err != nil {
		return nil, nil, errors.Wrap(err, "ExecuteValueAndGrad: value_and_grad apply")
	}

	evalArrays := make([]*bridge.Array, 0, len(valueArrays)+len(gradArrays))
	evalArrays = append(evalArrays, valueArrays...)
	evalArrays = append(evalArrays, gradArrays...)
	if err := bridge.Eval(evalArrays...); err != nil {
		for _, arr := range valueArrays {
			arr.Free()
		}
		for _, arr := range gradArrays {
			arr.Free()
		}
		return nil, nil, errors.Wrap(err, "ExecuteValueAndGrad: eval")
	}

	values = make([]backends.Buffer, len(e.outputShapes))
	for ii, shape := range e.outputShapes {
		values[ii] = newBufferFromArray(valueArrays[ii], shape)
	}

	grads = make([]backends.Buffer, len(e.gradShapes))
	for ii, shape := range e.gradShapes {
		grads[ii] = newBufferFromArray(gradArrays[ii], shape)
	}

	finalizeDonatedInputs(e.backend, inputs, donate)
	return values, grads, nil
}
