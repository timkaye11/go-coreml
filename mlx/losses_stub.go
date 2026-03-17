// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin || !cgo

package mlx

// InfoNCEResult holds the result of GPU-side InfoNCE computation.
type InfoNCEResult struct {
	Loss    float64
	DLdFlat []float32
}

// InfoNCEGradientMLX is a stub for non-darwin/non-cgo builds.
func InfoNCEGradientMLX(
	flatVecs []float32,
	chunkMaskFlat []float32,
	docIDs []int,
	N, dim int,
	temperature float64,
	focalGamma, focalAlpha float64,
) *InfoNCEResult {
	panic("InfoNCEGradientMLX: MLX losses require darwin with CGO")
}

// FLOPSLossMLX is a stub for non-darwin/non-cgo builds.
func FLOPSLossMLX(
	flat []float32,
	mask []float32,
	N, V int,
) (float64, []float32) {
	panic("FLOPSLossMLX: MLX losses require darwin with CGO")
}
