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

	// Create a Go closure that replays the tape and returns output arrays.
	// Note: we do NOT free intermediates here. mlx_compile traces this closure
	// once and caches the fused graph. On cache hits, MLX reuses the cached
	// graph and discards the lazy arrays we create, so freeing them is wasted
	// work. MLX's internal refcounting keeps the trace graph alive as needed.
	replayFn := func(inputs []*bridge.Array) []*bridge.Array {
		s := backend.stream()
		arrays := replayTape(mainFn, inputs, s)

		outputs := make([]*bridge.Array, len(mainFn.outputs))
		for i, out := range mainFn.outputs {
			outputs[i] = arrays[out.tapeIdx]
		}

		return outputs
	}

	// Wrap as an MLX closure and compile it for fused execution.
	rawClosure := bridge.NewClosureFromGoFunc(replayFn)
	compiled := bridge.CompileClosure(rawClosure, false)

	return &Executable{
		backend:      b.backend,
		mainFn:       b.mainFn,
		compiled:     compiled,
		rawClosure:   rawClosure,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: collectOutputShapes(b.mainFn.outputs),
	}, nil
}

// compileWithControlFlow compiles a function containing a control flow operation.
func (b *Builder) compileWithControlFlow() (backends.Executable, error) {
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
