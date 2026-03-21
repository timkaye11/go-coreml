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

// replayTape replays a function's tape with the given parameter arrays.
// Returns the arrays produced at each tape index.
func replayTape(fn *Function, paramArrays []*bridge.Array, s *bridge.Stream) []*bridge.Array {
	arrays := make([]*bridge.Array, len(fn.tape))
	// Fill in parameter slots.
	for i, param := range fn.params {
		arrays[param.tapeIdx] = paramArrays[i]
	}
	// Replay non-parameter tape entries.
	for i, tapeFn := range fn.tape {
		if tapeFn == nil {
			continue // parameter — already filled
		}
		arrays[i] = tapeFn(arrays, s)
	}
	return arrays
}

// freeIntermediates frees all replayed arrays except parameters, constants, and outputs.
func freeIntermediates(fn *Function, arrays []*bridge.Array, outputTapeIndices map[int]bool) {
	skip := make(map[int]bool)
	// Skip parameters (owned by caller).
	for _, p := range fn.params {
		skip[p.tapeIdx] = true
	}
	// Skip constants (shared with the Function, must not be freed).
	for _, idx := range fn.constIndices {
		skip[idx] = true
	}
	// Skip outputs (will be wrapped as buffers, caller frees them).
	for idx := range outputTapeIndices {
		skip[idx] = true
	}
	for i, arr := range arrays {
		if arr != nil && !skip[i] {
			arr.Free()
		}
	}
}

// Executable implements backends.Executable for the MLX backend.
// It uses a compiled MLX closure for fused GPU execution.
type Executable struct {
	backend      *Backend
	mainFn       *Function
	compiled     *bridge.Closure // compiled closure for fused execution
	rawClosure   *bridge.Closure // uncompiled closure (must be freed)
	inputNames   []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
	mode         ExecutionMode
	reasons      []string
	mu           sync.Mutex
}

var _ backends.Executable = &Executable{}

func (e *Executable) Finalize() {
	if e.compiled != nil {
		e.compiled.Free()
		e.compiled = nil
	}
	if e.rawClosure != nil {
		e.rawClosure.Free()
		e.rawClosure = nil
	}
	if e.mainFn != nil {
		e.mainFn.finalize()
	}
}

func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

func (e *Executable) ExecutionMode() ExecutionMode {
	return e.mode
}

func (e *Executable) ExecutionReasons() []string {
	if len(e.reasons) == 0 {
		return nil
	}
	return append([]string(nil), e.reasons...)
}

func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) ([]backends.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	// Extract input arrays.
	paramArrays := make([]*bridge.Array, len(inputs))
	for i, input := range inputs {
		buf, ok := input.(*mlxBuffer)
		if !ok {
			return nil, errors.Errorf("Execute: input #%d is not a *mlxBuffer, got %T", i, input)
		}
		if !buf.shape.Equal(e.inputShapes[i]) {
			return nil, errors.Errorf("Execute: input #%d shape mismatch: expected %s, got %s",
				i, e.inputShapes[i], buf.shape)
		}
		paramArrays[i] = buf.array
	}

	// Use compiled closure for fused GPU execution.
	outputArrays, err := bridge.ApplyClosure(e.compiled, paramArrays)
	if err != nil {
		return nil, errors.Wrap(err, "Execute: compiled closure apply")
	}

	// Evaluate output arrays.
	if err := bridge.Eval(outputArrays...); err != nil {
		for _, a := range outputArrays {
			a.Free()
		}
		return nil, errors.Wrap(err, "Execute: eval")
	}

	// Wrap output arrays as mlxBuffers.
	results := make([]backends.Buffer, len(e.outputShapes))
	for i, out := range e.mainFn.outputs {
		results[i] = newBufferFromArray(outputArrays[i], out.shape)
	}
	finalizeDonatedInputs(e.backend, inputs, donate)

	return results, nil
}

// evalFunctionOutputs replays fn's tape with the given param arrays, evaluates the
// output arrays, wraps them as mlxBuffers, frees intermediates, and returns the results.
// It is shared by the execIf and execCall control flow handlers.
func (e *ExecutableWithCF) evalFunctionOutputs(fn *Function, paramArrays []*bridge.Array, errCtx string) ([]backends.Buffer, error) {
	s := e.backend.stream()
	arrays := replayTape(fn, paramArrays, s)

	outputTapeIndices := make(map[int]bool, len(fn.outputs))
	outs := make([]*bridge.Array, len(fn.outputs))
	for i, out := range fn.outputs {
		outs[i] = arrays[out.tapeIdx]
		outputTapeIndices[out.tapeIdx] = true
	}
	if err := bridge.Eval(outs...); err != nil {
		freeIntermediates(fn, arrays, outputTapeIndices)
		return nil, errors.Wrap(err, errCtx)
	}

	results := make([]backends.Buffer, len(fn.outputs))
	for i, out := range fn.outputs {
		results[i] = newBufferFromArray(arrays[out.tapeIdx], out.shape)
	}
	freeIntermediates(fn, arrays, outputTapeIndices)
	return results, nil
}

// ===========================================================================
// Control Flow Executable
// ===========================================================================

type outputSource struct {
	fromCF  bool
	cfIndex int
	preNode *graphNode
}

type ExecutableWithCF struct {
	backend       *Backend
	mainFn        *Function
	cfStep        *controlFlowStep
	outputMapping []outputSource
	inputNames    []string
	inputShapes   []shapes.Shape
	outputShapes  []shapes.Shape
	mode          ExecutionMode
	reasons       []string
	mu            sync.Mutex
}

var _ backends.Executable = &ExecutableWithCF{}

func (e *ExecutableWithCF) Finalize() {
	if e.mainFn != nil {
		e.mainFn.finalize()
	}
}

func (e *ExecutableWithCF) Inputs() ([]string, []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

func (e *ExecutableWithCF) Outputs() []shapes.Shape {
	return e.outputShapes
}

func (e *ExecutableWithCF) ExecutionMode() ExecutionMode {
	return e.mode
}

func (e *ExecutableWithCF) ExecutionReasons() []string {
	if len(e.reasons) == 0 {
		return nil
	}
	return append([]string(nil), e.reasons...)
}

func (e *ExecutableWithCF) Execute(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) ([]backends.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("ExecuteWithCF: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	// Extract input arrays and replay tape.
	paramArrays := make([]*bridge.Array, len(inputs))
	for i, input := range inputs {
		buf := input.(*mlxBuffer)
		paramArrays[i] = buf.array
	}
	s := e.backend.stream()
	arrays := replayTape(e.mainFn, paramArrays, s)

	// Evaluate CF input arrays.
	cf := e.cfStep
	cfInputArrays := make([]*bridge.Array, len(cf.inputs))
	for i, node := range cf.inputs {
		cfInputArrays[i] = arrays[node.tapeIdx]
	}
	if err := bridge.Eval(cfInputArrays...); err != nil {
		return nil, errors.Wrap(err, "ExecuteWithCF: eval CF inputs")
	}

	// Execute control flow using replayed arrays.
	var cfResults []backends.Buffer
	var err error
	switch cf.opType {
	case backends.OpTypeWhile:
		cfResults, err = e.execWhile(cf, cfInputArrays)
	case backends.OpTypeIf:
		cfResults, err = e.execIf(cf, cfInputArrays)
	case backends.OpTypeSort:
		cfResults, err = e.execSort(cf, cfInputArrays)
	case backends.OpTypeCall:
		cfResults, err = e.execCall(cf, cfInputArrays)
	default:
		err = errors.Errorf("unsupported control flow op: %v", cf.opType)
	}
	if err != nil {
		return nil, err
	}

	// Assemble final outputs and track which tape indices are used as outputs.
	outputTapeIndices := make(map[int]bool)
	results := make([]backends.Buffer, len(e.outputMapping))
	for i, src := range e.outputMapping {
		if src.fromCF {
			results[i] = cfResults[src.cfIndex]
		} else {
			preArr := arrays[src.preNode.tapeIdx]
			outputTapeIndices[src.preNode.tapeIdx] = true
			if evalErr := bridge.Eval(preArr); evalErr != nil {
				return nil, errors.Wrap(evalErr, "ExecuteWithCF: eval pre-CF output")
			}
			results[i] = newBufferFromArray(preArr, src.preNode.shape)
		}
	}

	// Free intermediate arrays from tape replay that are not used as outputs.
	// Without this, all intermediate arrays from replayTape leak every step
	// (~1.8 MB/step for a 1-layer model with VJP gradients).
	freeIntermediates(e.mainFn, arrays, outputTapeIndices)
	finalizeDonatedInputs(e.backend, inputs, donate)

	return results, nil
}

func finalizeDonatedInputs(backend *Backend, inputs []backends.Buffer, donate []bool) {
	if backend == nil || len(inputs) == 0 || len(donate) == 0 {
		return
	}
	seen := make(map[*mlxBuffer]struct{}, len(inputs))
	for i, input := range inputs {
		if i >= len(donate) || !donate[i] || input == nil {
			continue
		}
		buf, ok := input.(*mlxBuffer)
		if !ok || buf == nil {
			continue
		}
		if _, found := seen[buf]; found {
			continue
		}
		seen[buf] = struct{}{}
		_ = backend.BufferFinalize(buf)
	}
}

func (e *ExecutableWithCF) execWhile(cf *controlFlowStep, cfInputArrays []*bridge.Array) ([]backends.Buffer, error) {
	wd := cf.whileData
	s := e.backend.stream()

	stateCount := len(cfInputArrays)
	state := make([]*bridge.Array, stateCount)
	stateShapes := make([]shapes.Shape, stateCount)
	for i, node := range cf.inputs {
		state[i] = cfInputArrays[i]
		stateShapes[i] = node.shape
	}

	const maxIterations = 1000000
	for iter := range maxIterations {
		// Replay condition tape with current state as params.
		condArrays := replayTape(wd.condFn, state, s)
		condOut := condArrays[wd.condFn.outputs[0].tapeIdx]
		if err := bridge.Eval(condOut); err != nil {
			return nil, errors.Wrapf(err, "While: cond eval iteration %d", iter)
		}
		condPtr := condOut.DataPtr()
		condValue := *(*bool)(condPtr)

		// Free condition intermediates (output is a scalar bool, already read).
		condOutputIndices := map[int]bool{wd.condFn.outputs[0].tapeIdx: true}
		freeIntermediates(wd.condFn, condArrays, condOutputIndices)
		condOut.Free()

		if !condValue {
			break
		}
		if iter == maxIterations-1 {
			return nil, errors.Errorf("While: exceeded maximum iterations (%d)", maxIterations)
		}

		// Replay body tape with current state as params.
		bodyArrays := replayTape(wd.bodyFn, state, s)
		bodyOuts := make([]*bridge.Array, len(wd.bodyFn.outputs))
		bodyOutputIndices := make(map[int]bool, len(wd.bodyFn.outputs))
		for i, out := range wd.bodyFn.outputs {
			bodyOuts[i] = bodyArrays[out.tapeIdx]
			bodyOutputIndices[out.tapeIdx] = true
		}
		if err := bridge.Eval(bodyOuts...); err != nil {
			return nil, errors.Wrapf(err, "While: body eval iteration %d", iter)
		}

		// Free body intermediates (outputs become next state).
		freeIntermediates(wd.bodyFn, bodyArrays, bodyOutputIndices)

		// Collect new state from body outputs.
		newState := make([]*bridge.Array, stateCount)
		for i, out := range wd.bodyFn.outputs {
			newState[i] = bodyArrays[out.tapeIdx]
		}

		// Free previous state arrays (skip iter 0: those are caller-owned cfInputArrays).
		// Avoid freeing arrays that are reused as new state (identity passthrough).
		if iter > 0 {
			for i, arr := range state {
				if arr != nil && arr != newState[i] {
					arr.Free()
				}
			}
		}

		// Update state.
		copy(state, newState)
	}

	results := make([]backends.Buffer, stateCount)
	for i := range state {
		results[i] = newBufferFromArray(state[i], stateShapes[i])
	}
	return results, nil
}

func (e *ExecutableWithCF) execIf(cf *controlFlowStep, cfInputArrays []*bridge.Array) ([]backends.Buffer, error) {
	id := cf.ifData
	predValue := *(*bool)(cfInputArrays[0].DataPtr())

	branchFn := id.falseFn
	if predValue {
		branchFn = id.trueFn
	}
	// Branch closures have no params; replay with nil.
	return e.evalFunctionOutputs(branchFn, nil, "If: branch eval")
}

func (e *ExecutableWithCF) execSort(cf *controlFlowStep, cfInputArrays []*bridge.Array) ([]backends.Buffer, error) {
	sd := cf.sortData
	s := e.backend.stream()

	if len(cfInputArrays) == 1 {
		sorted := bridge.Sort(cfInputArrays[0], sd.axis, s)
		if err := bridge.Eval(sorted); err != nil {
			return nil, errors.Wrap(err, "Sort: eval")
		}
		return []backends.Buffer{newBufferFromArray(sorted, cf.inputs[0].shape)}, nil
	}

	indices := bridge.ArgSort(cfInputArrays[0], sd.axis, s)
	if err := bridge.Eval(indices); err != nil {
		return nil, errors.Wrap(err, "Sort: argsort eval")
	}

	results := make([]backends.Buffer, len(cfInputArrays))
	for i, arr := range cfInputArrays {
		sorted := bridge.TakeAlongAxis(arr, indices, sd.axis, s)
		if err := bridge.Eval(sorted); err != nil {
			return nil, errors.Wrapf(err, "Sort: take_along_axis input %d", i)
		}
		results[i] = newBufferFromArray(sorted, cf.inputs[i].shape)
	}
	indices.Free()
	return results, nil
}

func (e *ExecutableWithCF) execCall(cf *controlFlowStep, cfInputArrays []*bridge.Array) ([]backends.Buffer, error) {
	return e.evalFunctionOutputs(cf.callData.targetFn, cfInputArrays, "Call: eval")
}
