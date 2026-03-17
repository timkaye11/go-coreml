// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"math"
	"testing"
)

// cpuInfoNCEGradient is a reference implementation matching infonce_cpu.go.
func cpuInfoNCEGradient(
	flatVecs []float32,
	chunkMaskFlat []float32,
	docIDs []int,
	N, dim int,
	temperature float64,
	focalGamma, focalAlpha float64,
) (float64, []float32) {
	validIdx := make([]int, 0, N)
	for i := 0; i < N; i++ {
		if chunkMaskFlat[i] > 0.5 {
			validIdx = append(validIdx, i)
		}
	}
	V := len(validIdx)
	if V < 2 {
		return 0, make([]float32, N*dim)
	}

	norms := make([]float64, V)
	normFlat := make([]float32, V*dim)
	for ci, origI := range validIdx {
		var sumSq float64
		for j := 0; j < dim; j++ {
			val := float64(flatVecs[origI*dim+j])
			sumSq += val * val
		}
		norms[ci] = math.Sqrt(sumSq + 1e-12)
		invNorm := 1.0 / norms[ci]
		for j := 0; j < dim; j++ {
			normFlat[ci*dim+j] = float32(float64(flatVecs[origI*dim+j]) * invNorm)
		}
	}

	// Similarity
	sim := make([]float32, V*V)
	for i := 0; i < V; i++ {
		for j := i; j < V; j++ {
			var dot float64
			for k := 0; k < dim; k++ {
				dot += float64(normFlat[i*dim+k]) * float64(normFlat[j*dim+k])
			}
			sim[i*V+j] = float32(dot)
			sim[j*V+i] = float32(dot)
		}
	}
	invT := 1.0 / temperature
	for i := range sim {
		sim[i] = float32(float64(sim[i]) * invT)
	}

	compactDocIDs := make([]int, V)
	for ci, origI := range validIdx {
		compactDocIDs[ci] = docIDs[origI]
	}
	posMask := make([]float32, V*V)
	posCount := make([]float32, V)
	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			if i != j && compactDocIDs[i] == compactDocIDs[j] {
				posMask[i*V+j] = 1.0
				posCount[i]++
			}
		}
	}

	dLdSim := make([]float64, V*V)
	loss := 0.0
	numAnchors := 0
	for i := 0; i < V; i++ {
		if posCount[i] == 0 {
			continue
		}
		numAnchors++
		invPosCount := 1.0 / float64(posCount[i])
		maxSim := -1e30
		for j := 0; j < V; j++ {
			if j != i && float64(sim[i*V+j]) > maxSim {
				maxSim = float64(sim[i*V+j])
			}
		}
		var sumExp float64
		for j := 0; j < V; j++ {
			if j != i {
				sumExp += math.Exp(float64(sim[i*V+j]) - maxSim)
			}
		}
		logSumExp := maxSim + math.Log(sumExp+1e-30)
		var meanPosSim float64
		for j := 0; j < V; j++ {
			if posMask[i*V+j] > 0.5 {
				meanPosSim += float64(sim[i*V+j])
			}
		}
		meanPosSim *= invPosCount
		loss += -meanPosSim + logSumExp
		for j := 0; j < V; j++ {
			if j == i {
				continue
			}
			softmaxJ := math.Exp(float64(sim[i*V+j])-maxSim) / sumExp
			dLdSim[i*V+j] = softmaxJ
			if posMask[i*V+j] > 0.5 {
				dLdSim[i*V+j] -= invPosCount
			}
		}
	}
	if numAnchors > 0 {
		loss /= float64(numAnchors)
		scale := 1.0 / float64(numAnchors)
		for i := range dLdSim {
			dLdSim[i] *= scale
		}
	}

	dLdSimSym := make([]float64, V*V)
	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			dLdSimSym[i*V+j] = dLdSim[i*V+j] + dLdSim[j*V+i]
		}
	}

	dLdNormFlat := make([]float64, V*dim)
	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			if dLdSimSym[i*V+j] == 0 {
				continue
			}
			s := invT * dLdSimSym[i*V+j]
			for k := 0; k < dim; k++ {
				dLdNormFlat[i*dim+k] += s * float64(normFlat[j*dim+k])
			}
		}
	}

	dLdFlat := make([]float32, N*dim)
	for ci := 0; ci < V; ci++ {
		var dotProd float64
		for j := 0; j < dim; j++ {
			dotProd += float64(normFlat[ci*dim+j]) * dLdNormFlat[ci*dim+j]
		}
		invNorm := 1.0 / norms[ci]
		origBase := validIdx[ci] * dim
		for j := 0; j < dim; j++ {
			dLdFlat[origBase+j] = float32((dLdNormFlat[ci*dim+j] - float64(normFlat[ci*dim+j])*dotProd) * invNorm)
		}
	}

	return loss, dLdFlat
}

func TestInfoNCEGradientMLX(t *testing.T) {
	// 4 docs × 3 chunks/doc = 12 valid, dim=32
	N, dim := 16, 32
	flatVecs := make([]float32, N*dim)
	mask := make([]float32, N)
	docIDs := make([]int, N)

	for i := 0; i < 12; i++ {
		mask[i] = 1.0
		docIDs[i] = i / 3
		for j := 0; j < dim; j++ {
			flatVecs[i*dim+j] = float32(math.Sin(float64(i*dim+j)*0.1)) * 0.5
		}
	}
	// last 4 are padding (mask=0)

	cpuLoss, cpuGrad := cpuInfoNCEGradient(flatVecs, mask, docIDs, N, dim, 0.07, 0, 0)
	mlxResult := InfoNCEGradientMLX(flatVecs, mask, docIDs, N, dim, 0.07, 0, 0)

	// Loss comparison
	lossDiff := math.Abs(cpuLoss - mlxResult.Loss)
	if lossDiff > 1e-3 {
		t.Fatalf("loss mismatch: cpu=%g mlx=%g diff=%g", cpuLoss, mlxResult.Loss, lossDiff)
	}
	t.Logf("Loss: cpu=%g mlx=%g diff=%g", cpuLoss, mlxResult.Loss, lossDiff)

	// Gradient comparison (float32 precision + chained ops = looser tolerance)
	maxDiff := float32(0)
	maxRelDiff := float32(0)
	for i := range cpuGrad {
		diff := abs32(cpuGrad[i] - mlxResult.DLdFlat[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		denom := max32(abs32(cpuGrad[i]), 1e-8)
		rel := diff / denom
		if rel > maxRelDiff {
			maxRelDiff = rel
		}
	}
	t.Logf("Max abs diff: %g, max rel diff: %g", maxDiff, maxRelDiff)

	// Allow larger tolerance due to float32 matmul precision differences
	if maxRelDiff > 0.05 {
		t.Fatalf("gradient too different: max relative diff = %g", maxRelDiff)
	}
}

func TestFLOPSLossMLX(t *testing.T) {
	N, V := 8, 64
	flat := make([]float32, N*V)
	mask := make([]float32, N)

	for i := 0; i < 6; i++ {
		mask[i] = 1.0
		for j := 0; j < V; j++ {
			flat[i*V+j] = float32(math.Sin(float64(i*V+j)*0.01)) * 0.1
		}
	}

	// CPU reference
	validCount := 6
	invN := 1.0 / float64(validCount)
	meanAct := make([]float64, V)
	for i := 0; i < N; i++ {
		if mask[i] < 0.5 {
			continue
		}
		for j := 0; j < V; j++ {
			meanAct[j] += float64(flat[i*V+j])
		}
	}
	cpuLoss := 0.0
	for j := 0; j < V; j++ {
		meanAct[j] *= invN
		cpuLoss += meanAct[j] * meanAct[j]
	}

	mlxLoss, mlxGrad := FLOPSLossMLX(flat, mask, N, V)

	lossDiff := math.Abs(cpuLoss - mlxLoss)
	t.Logf("FLOPS loss: cpu=%g mlx=%g diff=%g", cpuLoss, mlxLoss, lossDiff)
	if lossDiff > 1e-4 {
		t.Fatalf("FLOPS loss mismatch: diff=%g", lossDiff)
	}

	// Verify gradient shape
	if len(mlxGrad) != N*V {
		t.Fatalf("gradient length: got %d want %d", len(mlxGrad), N*V)
	}

	// Verify padding rows have zero gradient
	for i := 6; i < N; i++ {
		for j := 0; j < V; j++ {
			if mlxGrad[i*V+j] != 0 {
				t.Fatalf("padding grad[%d][%d] = %g, want 0", i, j, mlxGrad[i*V+j])
			}
		}
	}
}
