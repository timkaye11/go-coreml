// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// GPU-accelerated loss computations using MLX Metal kernels.
// These operate on flat float32 slices and return both loss and gradients.

package mlx

import (
	"math"
	"runtime"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
)

const epsilonNorm = 1e-12

// InfoNCEResult holds the result of GPU-side InfoNCE computation.
type InfoNCEResult struct {
	Loss    float64
	DLdFlat []float32 // gradient [N*dim]
}

// InfoNCEGradientMLX computes InfoNCE contrastive loss on GPU via MLX.
//
// The expensive operations (L2 norm, similarity matmul, softmax, matmul backprop)
// run as vectorized Metal kernels. Chunk compaction and positive mask construction
// stay on CPU (index manipulation, not compute-bound).
//
// For small V (< ~50), the CPU version is faster due to kernel dispatch overhead.
// This function becomes beneficial at larger V (> 100) or when used in-graph (Phase 3).
func InfoNCEGradientMLX(
	flatVecs []float32,
	chunkMaskFlat []float32,
	docIDs []int,
	N, dim int,
	temperature float64,
	focalGamma, focalAlpha float64,
) *InfoNCEResult {
	// 1. Build valid index mapping on CPU (index manipulation, not compute)
	validIdx := make([]int, 0, N)
	for i := 0; i < N; i++ {
		if chunkMaskFlat[i] > 0.5 {
			validIdx = append(validIdx, i)
		}
	}
	V := len(validIdx)
	if V < 2 {
		return &InfoNCEResult{Loss: 0, DLdFlat: make([]float32, N*dim)}
	}

	// 2. Compact valid vectors into [V, dim] on CPU
	compactFlat := make([]float32, V*dim)
	for ci, origI := range validIdx {
		copy(compactFlat[ci*dim:(ci+1)*dim], flatVecs[origI*dim:(origI+1)*dim])
	}

	s := bridge.DefaultGPUStream()
	defer s.Free()

	// 3. GPU: L2 normalize
	vecArr := bridge.NewArrayFromData(unsafe.Pointer(&compactFlat[0]), []int{V, dim}, bridge.DTypeFloat32)
	defer vecArr.Free()

	eps := bridge.NewArrayScalarFloat32(float32(epsilonNorm))
	defer eps.Free()

	// squaredSum = sum(x², axis=-1, keepdims=true) → [V, 1]
	sqVec := bridge.Multiply(vecArr, vecArr, s)
	defer sqVec.Free()
	squaredSum := bridge.Sum(sqVec, []int{1}, true, s)
	defer squaredSum.Free()

	// norm = sqrt(squaredSum + eps) → [V, 1]
	normSq := bridge.Add(squaredSum, eps, s)
	defer normSq.Free()
	norm := bridge.Sqrt(normSq, s)
	defer norm.Free()

	// normalized = x / norm → [V, dim]
	normalized := bridge.Divide(vecArr, norm, s)
	defer normalized.Free()

	// 4. GPU: Similarity matrix = normalized @ normalized^T → [V, V]
	normalizedT := bridge.Transpose(normalized, nil, s)
	defer normalizedT.Free()
	simRaw := bridge.MatMul(normalized, normalizedT, s)
	defer simRaw.Free()

	// 5. Temperature scaling
	invT := bridge.NewArrayScalarFloat32(float32(1.0 / temperature))
	defer invT.Free()
	sim := bridge.Multiply(simRaw, invT, s)
	defer sim.Free()

	// 6. Read sim matrix to CPU for InfoNCE loss + gradient computation
	// (This is the softmax + mask logic that's hard to vectorize efficiently
	// due to variable positive counts per anchor and self-exclusion)
	simFlat := make([]float32, V*V)
	readFloat32Into(sim, simFlat)

	// Also read normalized vectors and norms for backprop
	normFlat := make([]float32, V*dim)
	readFloat32Into(normalized, normFlat)

	// norm is [V,1], reshape to [V] for readback
	normReshaped := bridge.Reshape(norm, []int{V}, s)
	defer normReshaped.Free()
	normVals := make([]float32, V)
	readFloat32Into(normReshaped, normVals)

	// 7. CPU: Build positive mask and compute InfoNCE loss + dL/dSim
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

	dLdSim := make([]float32, V*V)
	loss := 0.0
	numAnchors := 0

	for i := 0; i < V; i++ {
		if posCount[i] == 0 {
			continue
		}
		numAnchors++
		invPosCount := 1.0 / float64(posCount[i])

		maxSim := float64(-1e30)
		for j := 0; j < V; j++ {
			if j != i && float64(simFlat[i*V+j]) > maxSim {
				maxSim = float64(simFlat[i*V+j])
			}
		}

		var sumExp float64
		for j := 0; j < V; j++ {
			if j != i {
				sumExp += math.Exp(float64(simFlat[i*V+j]) - maxSim)
			}
		}
		logSumExp := maxSim + math.Log(sumExp+1e-30)

		var meanPosSim float64
		for j := 0; j < V; j++ {
			if posMask[i*V+j] > 0.5 {
				meanPosSim += float64(simFlat[i*V+j])
			}
		}
		meanPosSim *= invPosCount

		focalWeight := 1.0
		if focalGamma > 0 {
			var meanPosSoftmax float64
			for j := 0; j < V; j++ {
				if j != i && posMask[i*V+j] > 0.5 {
					meanPosSoftmax += math.Exp(float64(simFlat[i*V+j])-maxSim) / sumExp
				}
			}
			meanPosSoftmax *= invPosCount
			focalWeight = math.Pow(1.0-meanPosSoftmax, focalGamma) * focalAlpha
		}

		loss += focalWeight * (-meanPosSim + logSumExp)

		for j := 0; j < V; j++ {
			if j == i {
				continue
			}
			softmaxJ := math.Exp(float64(simFlat[i*V+j])-maxSim) / sumExp
			val := focalWeight * softmaxJ
			if posMask[i*V+j] > 0.5 {
				val -= focalWeight * invPosCount
			}
			dLdSim[i*V+j] = float32(val)
		}
	}

	if numAnchors > 0 {
		loss /= float64(numAnchors)
		scale := float32(1.0 / float64(numAnchors))
		for i := range dLdSim {
			dLdSim[i] *= scale
		}
	}

	// 8. Symmetrize
	dLdSimSym := make([]float32, V*V)
	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			dLdSimSym[i*V+j] = dLdSim[i*V+j] + dLdSim[j*V+i]
		}
	}

	// 9. GPU: Backprop through matmul → dL/dNorm = (1/τ) * dL/dSimSym @ normFlat
	dLdSimArr := bridge.NewArrayFromData(unsafe.Pointer(&dLdSimSym[0]), []int{V, V}, bridge.DTypeFloat32)
	defer dLdSimArr.Free()
	normArr := bridge.NewArrayFromData(unsafe.Pointer(&normFlat[0]), []int{V, dim}, bridge.DTypeFloat32)
	defer normArr.Free()

	scaledDLdSim := bridge.Multiply(invT, dLdSimArr, s)
	defer scaledDLdSim.Free()
	dLdNorm := bridge.MatMul(scaledDLdSim, normArr, s)
	defer dLdNorm.Free()

	// 10. GPU: Backprop through L2 norm
	// dL/dx = (dL/dy - y * sum(y * dL/dy, axis=-1, keepdims)) / ||x||
	yDotGrad := bridge.Multiply(normArr, dLdNorm, s)
	defer yDotGrad.Free()
	dotSum := bridge.Sum(yDotGrad, []int{1}, true, s)
	defer dotSum.Free()
	yDot := bridge.Multiply(normArr, dotSum, s)
	defer yDot.Free()
	diff := bridge.Subtract(dLdNorm, yDot, s)
	defer diff.Free()

	normExpanded := bridge.NewArrayFromData(unsafe.Pointer(&normVals[0]), []int{V, 1}, bridge.DTypeFloat32)
	defer normExpanded.Free()
	dLdX := bridge.Divide(diff, normExpanded, s)
	defer dLdX.Free()

	// 11. Read back and scatter to original layout
	dLdCompact := make([]float32, V*dim)
	readFloat32Into(dLdX, dLdCompact)

	dLdFlat := make([]float32, N*dim)
	for ci, origI := range validIdx {
		copy(dLdFlat[origI*dim:(origI+1)*dim], dLdCompact[ci*dim:(ci+1)*dim])
	}

	runtime.KeepAlive(flatVecs)
	runtime.KeepAlive(compactFlat)
	runtime.KeepAlive(dLdSimSym)
	runtime.KeepAlive(normFlat)
	runtime.KeepAlive(normVals)

	return &InfoNCEResult{Loss: loss, DLdFlat: dLdFlat}
}

// FLOPSLossMLX computes FLOPS regularization loss on GPU via MLX.
// FLOPS loss = sum_j( mean_i(act_{i,j})^2 )
//
// For vocab_size=50368 and V=25 valid chunks, this is a meaningful speedup
// over the CPU version since the mean+square operations are vectorized.
func FLOPSLossMLX(
	flat []float32,
	mask []float32,
	N, V int,
) (float64, []float32) {
	validCount := 0
	for i := 0; i < N; i++ {
		if mask[i] > 0.5 {
			validCount++
		}
	}
	if validCount == 0 {
		return 0, make([]float32, N*V)
	}

	// Compact valid rows
	compact := make([]float32, validCount*V)
	ci := 0
	validIdx := make([]int, 0, validCount)
	for i := 0; i < N; i++ {
		if mask[i] > 0.5 {
			copy(compact[ci*V:(ci+1)*V], flat[i*V:(i+1)*V])
			validIdx = append(validIdx, i)
			ci++
		}
	}

	s := bridge.DefaultGPUStream()
	defer s.Free()

	// GPU: mean across valid samples → [V]
	arr := bridge.NewArrayFromData(unsafe.Pointer(&compact[0]), []int{validCount, V}, bridge.DTypeFloat32)
	defer arr.Free()

	invN := bridge.NewArrayScalarFloat32(float32(1.0 / float64(validCount)))
	defer invN.Free()

	// meanAct = sum(arr, axis=0) / validCount → [V]
	sumAct := bridge.Sum(arr, []int{0}, false, s)
	defer sumAct.Free()
	meanAct := bridge.Multiply(sumAct, invN, s)
	defer meanAct.Free()

	// loss = sum(meanAct²)
	meanSq := bridge.Multiply(meanAct, meanAct, s)
	defer meanSq.Free()
	lossArr := bridge.Sum(meanSq, []int{0}, false, s)
	defer lossArr.Free()

	// grad for each valid sample: 2 * meanAct / validCount → broadcast to [validCount, V]
	two := bridge.NewArrayScalarFloat32(2.0)
	defer two.Free()
	gradRow := bridge.Multiply(two, bridge.Multiply(meanAct, invN, s), s)
	defer gradRow.Free()

	// Read loss
	lossData := make([]float32, 1)
	readFloat32Into(lossArr, lossData)
	loss := float64(lossData[0])

	// Read gradient row and broadcast to all valid samples
	gradRowData := make([]float32, V)
	readFloat32Into(gradRow, gradRowData)

	grad := make([]float32, N*V)
	for _, origI := range validIdx {
		copy(grad[origI*V:(origI+1)*V], gradRowData)
	}

	runtime.KeepAlive(flat)
	runtime.KeepAlive(compact)

	return loss, grad
}
