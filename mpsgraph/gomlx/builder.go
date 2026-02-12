// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package mpsgraph

import (
	"github.com/gomlx/go-coreml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Builder implements backends.Builder for the MPSGraph backend.
type Builder struct {
	notimplemented.Builder // Bootstrap: unimplemented ops return ErrNotImplemented.

	backend  *Backend
	name     string
	ctx      *bridge.Context
	mainFn   *Function
	compiled bool
}

// Verify interface compliance.
var _ backends.Builder = &Builder{}

// newBuilder creates a builder with its own MPSGraph context.
func newBuilder(backend *Backend, name string, ctx *bridge.Context) *Builder {
	return &Builder{
		backend: backend,
		name:    name,
		ctx:     ctx,
	}
}

// Name returns the builder name.
func (b *Builder) Name() string { return b.name }

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

	// Gather feeds (parameters/placeholders) and targets (outputs).
	info := bridge.CompileInfo{
		Feeds:      make([]bridge.Tensor, len(b.mainFn.params)),
		FeedDtypes: make([]int, len(b.mainFn.params)),
		FeedShapes: make([][]int64, len(b.mainFn.params)),
		Targets:    make([]bridge.Tensor, len(b.mainFn.outputs)),
	}

	for i, p := range b.mainFn.params {
		info.Feeds[i] = p.tensor
		info.FeedDtypes[i] = dtypeToBridgeDType(p.shape.DType)
		dims := p.shape.Dimensions
		info.FeedShapes[i] = make([]int64, len(dims))
		for j, d := range dims {
			info.FeedShapes[i][j] = int64(d)
		}
	}

	for i, out := range b.mainFn.outputs {
		info.Targets[i] = out.tensor
	}

	exec, err := b.ctx.Compile(info)
	if err != nil {
		return nil, errors.Wrap(err, "compiling MPSGraph")
	}

	b.compiled = true

	inputNames := make([]string, len(b.mainFn.params))
	inputShapes := make([]shapes.Shape, len(b.mainFn.params))
	for i, p := range b.mainFn.params {
		inputNames[i] = p.name
		inputShapes[i] = p.shape
	}

	outputShapes := make([]shapes.Shape, len(b.mainFn.outputs))
	for i, out := range b.mainFn.outputs {
		outputShapes[i] = out.shape
	}

	return &Executable{
		backend:      b.backend,
		exec:         exec,
		inputNames:   inputNames,
		inputShapes:  inputShapes,
		outputShapes: outputShapes,
	}, nil
}
