// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"github.com/gomlx/go-coreml/mlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Builder implements backends.Builder for the MLX backend.
type Builder struct {
	notimplemented.Builder // Bootstrap: unimplemented ops return ErrNotImplemented.

	backend  *Backend
	name     string
	mainFn   *Function
	compiled bool
}

// Verify interface compliance.
var _ backends.Builder = &Builder{}

// newBuilder creates a builder.
func newBuilder(backend *Backend, name string) *Builder {
	return &Builder{
		backend: backend,
		name:    name,
	}
}

// Name returns the builder name.
func (b *Builder) Name() string { return b.name }

// Finalize is a no-op for MLX (lazy evaluation, no explicit graph context to free).
func (b *Builder) Finalize() {}

// Main returns the main function, creating it lazily.
func (b *Builder) Main() backends.Function {
	if b.mainFn == nil {
		b.mainFn = newFunction(b, "main", nil)
	}
	return b.mainFn
}

// NewFunction creates a named sub-function.
func (b *Builder) NewFunction(name string) (backends.Function, error) {
	return newFunction(b, name, nil), nil
}

// OpShape returns the shape of a computation graph value.
func (b *Builder) OpShape(op backends.Value) (shapes.Shape, error) {
	node, ok := op.(*graphNode)
	if !ok {
		return shapes.Invalid(), errors.Errorf("OpShape: expected *graphNode, got %T", op)
	}
	return node.shape, nil
}

// Compile compiles the computation graph into an executable.
func (b *Builder) Compile() (backends.Executable, error) {
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

	// If the main function has a control flow step, use CF-aware compilation.
	if b.mainFn.controlFlowStep != nil {
		return b.compileWithControlFlow()
	}

	return b.compileSimple()
}

// compileSimple compiles a function without control flow into a single Executable.
// It wraps the tape replay in an MLX compiled closure for fused GPU execution.
func (b *Builder) compileSimple() (backends.Executable, error) {
	inputNames, inputShapes := collectParamInfo(b.mainFn.params)
	mainFn := b.mainFn
	backend := b.backend
	reasons := mainFn.goCallbackReasons()

	// If the tape is fully C-interpretable, use the pure-C interpreter path
	// to eliminate CGo boundary crossings during steady-state execution.
	if !mainFn.hasGoCallback && len(mainFn.instrs) > 0 {
		mode := ExecutionModeCTapeInterpreter
		if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
			return nil, err
		}
		backend.logExecutionMode(b.name, mode, reasons)
		instrs, numSlots, outputSlots, constSlots, constArrays, paramSlots := mainFn.serializeTape()
		rawClosure := bridge.NewClosureFromCTape(instrs, numSlots, outputSlots, constSlots, constArrays, paramSlots)
		// Skip mlx_compile: the C tape interpreter already handles memory correctly
		// (frees intermediate slots, keeps output slots). mlx_compile adds graph fusion
		// but its internal cache leaks ~1.8 MB/step by retaining refs to input arrays.
		// The C tape interpreter is fast enough (~106ms/step) for training.
		return &Executable{
			backend:      b.backend,
			mainFn:       b.mainFn,
			compiled:     rawClosure,
			rawClosure:   nil, // rawClosure IS the compiled closure now
			inputNames:   inputNames,
			inputShapes:  inputShapes,
			outputShapes: collectOutputShapes(b.mainFn.outputs),
			mode:         mode,
			reasons:      reasons,
		}, nil
	}

	// Fallback: Go closure path (for ops that require Go callbacks).
	// Create a Go closure that replays the tape and returns output arrays.
	//
	// When compiled (mlx_compile): intermediates are NOT freed because mlx_compile
	// traces this closure once and caches the fused graph. On cache hits, MLX
	// reuses the cached graph and discards the lazy arrays we create.
	//
	// When NOT compiled (hasGoCallback=true): intermediates MUST be freed because
	// the raw closure is called every step. bridge.Array has no GC finalizer,
	// so without explicit Free(), intermediate MLX arrays leak permanently.
	needsFreeIntermediates := mainFn.hasGoCallback

	replayFn := func(inputs []*bridge.Array) []*bridge.Array {
		s := backend.stream()
		arrays := replayTape(mainFn, inputs, s)

		outputs := make([]*bridge.Array, len(mainFn.outputs))
		for i, out := range mainFn.outputs {
			outputs[i] = arrays[out.tapeIdx]
		}

		// Free intermediate arrays (NOT outputs — outputs are still lazy and
		// need to survive until bridge.Eval is called by the executor).
		// Outputs will be freed by goClosureCallback after NewVectorArray retains them.
		if needsFreeIntermediates {
			outputIndices := make(map[int]bool, len(mainFn.outputs))
			for _, out := range mainFn.outputs {
				outputIndices[out.tapeIdx] = true
			}
			freeIntermediates(mainFn, arrays, outputIndices)
		}

		return outputs
	}

	// Wrap as an MLX closure.
	rawClosure := bridge.NewClosureFromGoFunc(replayFn)

	// Only compile if there are no Go callbacks. mlx_compile traces the closure
	// with symbolic arrays, but Go callbacks (e.g. RNG) try to Eval and read
	// real data during the trace, causing a SIGSEGV.
	var compiled *bridge.Closure
	mode := ExecutionModeCompiledGoClosure
	if !mainFn.hasGoCallback {
		if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
			rawClosure.Free()
			return nil, err
		}
		compiled = bridge.CompileClosure(rawClosure, false)
	} else {
		mode = ExecutionModeRawGoClosure
		if err := backend.validateExecutionMode(b.name, mode, reasons); err != nil {
			rawClosure.Free()
			return nil, err
		}
		compiled = rawClosure
		rawClosure = nil // avoid double-free
	}
	backend.logExecutionMode(b.name, mode, reasons)

	return &Executable{
		backend:      b.backend,
		mainFn:       b.mainFn,
		compiled:     compiled,
		rawClosure:   rawClosure,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: collectOutputShapes(b.mainFn.outputs),
		mode:         mode,
		reasons:      reasons,
	}, nil
}

// compileWithControlFlow compiles a function containing a control flow operation.
func (b *Builder) compileWithControlFlow() (backends.Executable, error) {
	mode := ExecutionModeControlFlow
	reasons := []string{"control_flow"}
	if err := b.backend.validateExecutionMode(b.name, mode, reasons); err != nil {
		return nil, err
	}
	b.backend.logExecutionMode(b.name, mode, reasons)
	cf := b.mainFn.controlFlowStep

	// Build output mapping.
	outputMapping := make([]outputSource, len(b.mainFn.outputs))
	for i, out := range b.mainFn.outputs {
		cfIdx := indexOfNode(cf.outputNodes, out)
		if cfIdx >= 0 {
			outputMapping[i] = outputSource{fromCF: true, cfIndex: cfIdx}
		} else {
			outputMapping[i] = outputSource{fromCF: false, preNode: out}
		}
	}

	inputNames, inputShapes := collectParamInfo(b.mainFn.params)

	return &ExecutableWithCF{
		backend:       b.backend,
		mainFn:        b.mainFn,
		cfStep:        cf,
		outputMapping: outputMapping,
		inputNames:    inputNames,
		inputShapes:   inputShapes,
		outputShapes:  collectOutputShapes(b.mainFn.outputs),
		mode:          mode,
		reasons:       reasons,
	}, nil
}

// collectParamInfo extracts names and shapes from parameter nodes.
func collectParamInfo(params []*graphNode) ([]string, []shapes.Shape) {
	names := make([]string, len(params))
	paramShapes := make([]shapes.Shape, len(params))
	for i, p := range params {
		names[i] = p.name
		paramShapes[i] = p.shape
	}
	return names, paramShapes
}

// collectOutputShapes extracts shapes from output nodes.
func collectOutputShapes(outputs []*graphNode) []shapes.Shape {
	outShapes := make([]shapes.Shape, len(outputs))
	for i, out := range outputs {
		outShapes[i] = out.shape
	}
	return outShapes
}

// indexOfNode returns the index of a node in a slice, or -1.
func indexOfNode(nodes []*graphNode, n *graphNode) int {
	for i, node := range nodes {
		if node == n {
			return i
		}
	}
	return -1
}
