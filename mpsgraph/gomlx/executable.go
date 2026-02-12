// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package mpsgraph

import (
	"sync"

	"github.com/gomlx/go-coreml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Executable implements backends.Executable for the MPSGraph backend.
type Executable struct {
	backend      *Backend
	exec         *bridge.Exec
	inputNames   []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
	mu           sync.Mutex // Serialize execution for safety.
}

// Verify interface compliance.
var _ backends.Executable = &Executable{}

// Finalize releases the compiled executable.
func (e *Executable) Finalize() {
	if e.exec != nil {
		e.exec.Destroy()
		e.exec = nil
	}
}

// Inputs returns the parameter names and shapes.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

// Outputs returns the output shapes.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Execute runs the compiled graph with the given input buffers.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) ([]backends.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(inputs) != len(e.inputShapes) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.inputShapes), len(inputs))
	}

	// Prepare input data.
	execInputs := make([]bridge.ExecInput, len(inputs))
	for i, input := range inputs {
		buf, ok := input.(*gpuBuffer)
		if !ok {
			return nil, errors.Errorf("Execute: input #%d is not a *gpuBuffer, got %T", i, input)
		}
		if !buf.shape.Equal(e.inputShapes[i]) {
			return nil, errors.Errorf("Execute: input #%d shape mismatch: expected %s, got %s",
				i, e.inputShapes[i], buf.shape)
		}
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, buf.shape.Rank())
		for j, d := range buf.shape.Dimensions {
			dims[j] = int64(d)
		}
		execInputs[i] = bridge.ExecInput{
			Data:  ptr,
			Size:  nbytes,
			DType: dtypeToBridgeDType(buf.shape.DType),
			Shape: dims,
		}
	}

	// Prepare output buffers.
	outputBuffers := make([]*gpuBuffer, len(e.outputShapes))
	execOutputs := make([]bridge.ExecOutput, len(e.outputShapes))
	for i, outShape := range e.outputShapes {
		buf := newBuffer(outShape)
		outputBuffers[i] = buf
		ptr, nbytes := buf.flatDataPtr()
		dims := make([]int64, outShape.Rank())
		for j, d := range outShape.Dimensions {
			dims[j] = int64(d)
		}
		execOutputs[i] = bridge.ExecOutput{
			Data:  ptr,
			Size:  nbytes,
			DType: dtypeToBridgeDType(outShape.DType),
			Shape: dims,
		}
	}

	// Execute.
	if err := e.exec.Execute(execInputs, execOutputs); err != nil {
		return nil, errors.Wrap(err, "Execute")
	}

	// Convert to backends.Buffer interface.
	result := make([]backends.Buffer, len(outputBuffers))
	for i, buf := range outputBuffers {
		result[i] = buf
	}
	return result, nil
}
