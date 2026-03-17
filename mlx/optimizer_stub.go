// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin || !cgo

package mlx

// AdamWUpdateFloat32 is a stub for non-darwin/non-cgo builds.
func AdamWUpdateFloat32(grad, m, v, w []float32,
	lr, beta1, beta2, eps, wd, biasCorr1, biasCorr2 float64,
	cautious bool,
) {
	panic("AdamWUpdateFloat32: MLX optimizer requires darwin with CGO")
}

// AdamWBatchUpdateFloat32 is a stub for non-darwin/non-cgo builds.
func AdamWBatchUpdateFloat32(
	grads, ms, vs, ws [][]float32,
	lrs []float64,
	beta1, beta2, eps, wd, biasCorr1, biasCorr2 float64,
	cautious bool,
) {
	panic("AdamWBatchUpdateFloat32: MLX optimizer requires darwin with CGO")
}

// ScheduleFreeUpdateFloat32 is a stub for non-darwin/non-cgo builds.
func ScheduleFreeUpdateFloat32(gradData, z, v, x, y []float32,
	lr, beta1, beta2, eps, wd, biasCorr2, ck float64,
) {
	panic("ScheduleFreeUpdateFloat32: MLX optimizer requires darwin with CGO")
}
