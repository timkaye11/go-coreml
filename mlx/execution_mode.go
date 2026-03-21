// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import "fmt"

// ExecutionMode identifies how an MLX executable runs at steady state.
type ExecutionMode string

const (
	ExecutionModeCTapeInterpreter  ExecutionMode = "ctape_interpreter"
	ExecutionModeCompiledGoClosure ExecutionMode = "compiled_go_closure"
	ExecutionModeRawGoClosure      ExecutionMode = "raw_go_closure"
	ExecutionModeControlFlow       ExecutionMode = "control_flow"
)

// ExecutionModeReporter is implemented by MLX executables to expose their
// steady-state execution strategy.
type ExecutionModeReporter interface {
	ExecutionMode() ExecutionMode
	ExecutionReasons() []string
}

func (b *Backend) validateExecutionMode(name string, mode ExecutionMode, reasons []string) error {
	if b.config.StrictCTape && mode != ExecutionModeCTapeInterpreter {
		if len(reasons) == 0 {
			return fmt.Errorf("mlx: strict_ctape enabled, builder %q compiled to %s", name, mode)
		}
		return fmt.Errorf("mlx: strict_ctape enabled, builder %q compiled to %s (%v)", name, mode, reasons)
	}
	return nil
}

func (b *Backend) logExecutionMode(name string, mode ExecutionMode, reasons []string) {
	if !b.config.LogExecution {
		return
	}
	if len(reasons) == 0 {
		fmt.Printf("  MLX executable %q: %s\n", name, mode)
		return
	}
	fmt.Printf("  MLX executable %q: %s (%v)\n", name, mode, reasons)
}
