// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"fmt"
	"math"
	"testing"
)

// cpuAdamWUpdate is a reference implementation matching optimizer_cpu.go.
func cpuAdamWUpdate(grad, m, v, w []float32, lr, beta1, beta2, eps, wd, biasCorr1, biasCorr2 float64, cautious bool) {
	for j := range w {
		g := float64(grad[j])
		mj := beta1*float64(m[j]) + (1-beta1)*g
		vj := beta2*float64(v[j]) + (1-beta2)*g*g
		mHat := mj / biasCorr1
		vHat := vj / biasCorr2
		update := mHat / (math.Sqrt(vHat) + eps)
		if cautious && update*g < 0 {
			update = 0
		}
		wj := float64(w[j])
		wj = wj - lr*update - lr*wd*wj
		m[j] = float32(mj)
		v[j] = float32(vj)
		w[j] = float32(wj)
	}
}

// cpuScheduleFreeUpdate is a reference implementation matching optimizer_cpu.go.
func cpuScheduleFreeUpdate(gradData, z, v, x, y []float32, lr, beta1, beta2, eps, wd, biasCorr2, ck float64) {
	for j := range z {
		g := float64(gradData[j])
		zj := float64(z[j])
		vj := float64(v[j])
		xj := float64(x[j])
		yj := beta1*xj + (1-beta1)*zj
		zj -= lr * wd * yj
		vj = beta2*vj + (1-beta2)*g*g
		vHat := vj / biasCorr2
		denom := math.Sqrt(vHat) + eps
		zj -= lr * g / denom
		xj = (1-ck)*xj + ck*zj
		z[j] = float32(zj)
		v[j] = float32(vj)
		x[j] = float32(xj)
		y[j] = float32(beta1*xj + (1-beta1)*zj)
	}
}

func TestAdamWUpdateFloat32(t *testing.T) {
	const N = 1024
	lr, beta1, beta2, eps, wd := 1e-3, 0.9, 0.999, 1e-8, 0.01
	biasCorr1 := 1 - math.Pow(beta1, 5) // step 4 (0-indexed)
	biasCorr2 := 1 - math.Pow(beta2, 5)

	for _, cautious := range []bool{false, true} {
		t.Run(func() string {
			if cautious {
				return "cautious"
			}
			return "standard"
		}(), func(t *testing.T) {
			// Initialize with deterministic values
			gradCPU := make([]float32, N)
			mCPU := make([]float32, N)
			vCPU := make([]float32, N)
			wCPU := make([]float32, N)
			gradGPU := make([]float32, N)
			mGPU := make([]float32, N)
			vGPU := make([]float32, N)
			wGPU := make([]float32, N)

			for i := 0; i < N; i++ {
				g := float32(math.Sin(float64(i)*0.1)) * 0.01
				w := float32(math.Cos(float64(i)*0.07)) * 0.1
				gradCPU[i], gradGPU[i] = g, g
				wCPU[i], wGPU[i] = w, w
				// m, v start at zero (first step)
			}

			cpuAdamWUpdate(gradCPU, mCPU, vCPU, wCPU, lr, beta1, beta2, eps, wd, biasCorr1, biasCorr2, cautious)
			AdamWUpdateFloat32(gradGPU, mGPU, vGPU, wGPU, lr, beta1, beta2, eps, wd, biasCorr1, biasCorr2, cautious)

			// Compare: float32 ops have ~1e-6 relative error
			tol := float32(1e-5)
			for i := 0; i < N; i++ {
				if diff := abs32(mCPU[i] - mGPU[i]); diff > tol*max32(abs32(mCPU[i]), 1e-8) {
					t.Fatalf("m[%d] mismatch: cpu=%g gpu=%g diff=%g", i, mCPU[i], mGPU[i], diff)
				}
				if diff := abs32(vCPU[i] - vGPU[i]); diff > tol*max32(abs32(vCPU[i]), 1e-8) {
					t.Fatalf("v[%d] mismatch: cpu=%g gpu=%g diff=%g", i, vCPU[i], vGPU[i], diff)
				}
				if diff := abs32(wCPU[i] - wGPU[i]); diff > tol*max32(abs32(wCPU[i]), 1e-8) {
					t.Fatalf("w[%d] mismatch: cpu=%g gpu=%g diff=%g", i, wCPU[i], wGPU[i], diff)
				}
			}
		})
	}
}

func TestScheduleFreeUpdateFloat32(t *testing.T) {
	const N = 1024
	lr, beta1, beta2, eps, wd := 1e-3, 0.9, 0.999, 1e-8, 0.01
	biasCorr2 := 1 - math.Pow(beta2, 5)
	ck := 0.5

	gradCPU := make([]float32, N)
	zCPU := make([]float32, N)
	vCPU := make([]float32, N)
	xCPU := make([]float32, N)
	yCPU := make([]float32, N)

	gradGPU := make([]float32, N)
	zGPU := make([]float32, N)
	vGPU := make([]float32, N)
	xGPU := make([]float32, N)
	yGPU := make([]float32, N)

	for i := 0; i < N; i++ {
		g := float32(math.Sin(float64(i)*0.1)) * 0.01
		z := float32(math.Cos(float64(i)*0.07)) * 0.1
		x := float32(math.Sin(float64(i)*0.03)) * 0.05
		gradCPU[i], gradGPU[i] = g, g
		zCPU[i], zGPU[i] = z, z
		xCPU[i], xGPU[i] = x, x
	}

	cpuScheduleFreeUpdate(gradCPU, zCPU, vCPU, xCPU, yCPU, lr, beta1, beta2, eps, wd, biasCorr2, ck)
	ScheduleFreeUpdateFloat32(gradGPU, zGPU, vGPU, xGPU, yGPU, lr, beta1, beta2, eps, wd, biasCorr2, ck)

	// Tolerance is higher than AdamW because the CPU reference uses float64 intermediate
	// math while the GPU version uses float32 throughout. The chained operations in
	// Schedule-Free (more sequential dependencies) amplify the rounding difference.
	tol := float32(2e-4)
	for i := 0; i < N; i++ {
		if diff := abs32(zCPU[i] - zGPU[i]); diff > tol*max32(abs32(zCPU[i]), 1e-8) {
			t.Fatalf("z[%d] mismatch: cpu=%g gpu=%g", i, zCPU[i], zGPU[i])
		}
		if diff := abs32(vCPU[i] - vGPU[i]); diff > tol*max32(abs32(vCPU[i]), 1e-8) {
			t.Fatalf("v[%d] mismatch: cpu=%g gpu=%g", i, vCPU[i], vGPU[i])
		}
		if diff := abs32(xCPU[i] - xGPU[i]); diff > tol*max32(abs32(xCPU[i]), 1e-8) {
			t.Fatalf("x[%d] mismatch: cpu=%g gpu=%g", i, xCPU[i], xGPU[i])
		}
		if diff := abs32(yCPU[i] - yGPU[i]); diff > tol*max32(abs32(yCPU[i]), 1e-8) {
			t.Fatalf("y[%d] mismatch: cpu=%g gpu=%g", i, yCPU[i], yGPU[i])
		}
	}
}

func TestAdamWBatchUpdateFloat32(t *testing.T) {
	const N1, N2, N3 = 512, 1024, 768
	lr1, lr2, lr3 := 1e-3, 2e-3, 5e-4
	beta1, beta2, eps, wd := 0.9, 0.999, 1e-8, 0.01
	biasCorr1 := 1 - math.Pow(beta1, 3)
	biasCorr2 := 1 - math.Pow(beta2, 3)

	// Create 3 parameter groups with different sizes and LRs
	sizes := []int{N1, N2, N3}
	lrs := []float64{lr1, lr2, lr3}

	gradsCPU := make([][]float32, 3)
	msCPU := make([][]float32, 3)
	vsCPU := make([][]float32, 3)
	wsCPU := make([][]float32, 3)
	gradsGPU := make([][]float32, 3)
	msGPU := make([][]float32, 3)
	vsGPU := make([][]float32, 3)
	wsGPU := make([][]float32, 3)

	for p := 0; p < 3; p++ {
		n := sizes[p]
		gradsCPU[p] = make([]float32, n)
		msCPU[p] = make([]float32, n)
		vsCPU[p] = make([]float32, n)
		wsCPU[p] = make([]float32, n)
		gradsGPU[p] = make([]float32, n)
		msGPU[p] = make([]float32, n)
		vsGPU[p] = make([]float32, n)
		wsGPU[p] = make([]float32, n)
		for i := 0; i < n; i++ {
			g := float32(math.Sin(float64(i+p*1000)*0.1)) * 0.01
			w := float32(math.Cos(float64(i+p*1000)*0.07)) * 0.1
			gradsCPU[p][i], gradsGPU[p][i] = g, g
			wsCPU[p][i], wsGPU[p][i] = w, w
		}
	}

	// Run CPU reference per-parameter
	for p := 0; p < 3; p++ {
		cpuAdamWUpdate(gradsCPU[p], msCPU[p], vsCPU[p], wsCPU[p],
			lrs[p], beta1, beta2, eps, wd, biasCorr1, biasCorr2, false)
	}

	// Run batched GPU
	AdamWBatchUpdateFloat32(gradsGPU, msGPU, vsGPU, wsGPU,
		lrs, beta1, beta2, eps, wd, biasCorr1, biasCorr2, false)

	// Compare
	tol := float32(1e-5)
	for p := 0; p < 3; p++ {
		for i := range msCPU[p] {
			if diff := abs32(wsCPU[p][i] - wsGPU[p][i]); diff > tol*max32(abs32(wsCPU[p][i]), 1e-8) {
				t.Fatalf("param %d w[%d] mismatch: cpu=%g gpu=%g", p, i, wsCPU[p][i], wsGPU[p][i])
			}
		}
	}
}

func BenchmarkAdamWUpdateFloat32(b *testing.B) {
	// Realistic scenario: 28 LoRA params, each 768*16 = 12288 elements
	const nParams = 28
	const paramSize = 12288

	grads := make([][]float32, nParams)
	ms := make([][]float32, nParams)
	vs := make([][]float32, nParams)
	ws := make([][]float32, nParams)
	lrs := make([]float64, nParams)
	for p := 0; p < nParams; p++ {
		grads[p] = make([]float32, paramSize)
		ms[p] = make([]float32, paramSize)
		vs[p] = make([]float32, paramSize)
		ws[p] = make([]float32, paramSize)
		lrs[p] = 1e-3 * float64(p+1) / float64(nParams)
		for i := range grads[p] {
			grads[p][i] = float32(i+p*paramSize) * 0.0001
			ws[p][i] = float32(i+p*paramSize) * 0.001
		}
	}

	b.Run(fmt.Sprintf("CPU_28x%d", paramSize), func(b *testing.B) {
		for b.Loop() {
			for p := 0; p < nParams; p++ {
				cpuAdamWUpdate(grads[p], ms[p], vs[p], ws[p],
					lrs[p], 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999, false)
			}
		}
	})

	b.Run(fmt.Sprintf("GPU_per_param_28x%d", paramSize), func(b *testing.B) {
		for b.Loop() {
			for p := 0; p < nParams; p++ {
				AdamWUpdateFloat32(grads[p], ms[p], vs[p], ws[p],
					lrs[p], 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999, false)
			}
		}
	})

	b.Run(fmt.Sprintf("GPU_batched_28x%d", paramSize), func(b *testing.B) {
		for b.Loop() {
			AdamWBatchUpdateFloat32(grads, ms, vs, ws, lrs,
				0.9, 0.999, 1e-8, 0.01, 0.9, 0.999, false)
		}
	})
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
