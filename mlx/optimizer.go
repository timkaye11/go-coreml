// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// GPU-accelerated optimizer operations using MLX Metal kernels.
// All functions operate on flat []float32 slices for compatibility with
// the existing CPU optimizer interface.

package mlx

import (
	"runtime"
	"unsafe"

	"github.com/gomlx/go-coreml/mlx/internal/bridge"
)

// arrayFromFloat32 creates an MLX array from a flat float32 slice without copying
// (MLX uses unified memory on Apple Silicon — the data is accessible by both CPU and GPU).
func arrayFromFloat32(data []float32) *bridge.Array {
	return bridge.NewArrayFromData(
		unsafe.Pointer(&data[0]),
		[]int{len(data)},
		bridge.DTypeFloat32,
	)
}

// readFloat32Into evaluates arr and copies results into dst.
func readFloat32Into(arr *bridge.Array, dst []float32) {
	if err := bridge.Eval(arr); err != nil {
		panic("mlx optimizer: eval failed: " + err.Error())
	}
	src := arr.DataPtr()
	n := len(dst) * 4
	copy(
		unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), n),
		unsafe.Slice((*byte)(src), n),
	)
}

// AdamWUpdateFloat32 performs Cautious AdamW on the GPU via MLX vectorized ops.
// All slices (grad, m, v, w) must have the same length and are updated in-place.
//
// This is the GPU equivalent of optimizer_cpu.go:adamWUpdateInPlace.
// ~15 Metal kernel dispatches replace N scalar iterations.
func AdamWUpdateFloat32(grad, m, v, w []float32,
	lr, beta1, beta2, eps, wd, biasCorr1, biasCorr2 float64,
	cautious bool,
) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	gArr := arrayFromFloat32(grad)
	defer gArr.Free()
	mArr := arrayFromFloat32(m)
	defer mArr.Free()
	vArr := arrayFromFloat32(v)
	defer vArr.Free()
	wArr := arrayFromFloat32(w)
	defer wArr.Free()

	// Scalar constants
	b1 := bridge.NewArrayScalarFloat32(float32(beta1))
	defer b1.Free()
	oneMinusB1 := bridge.NewArrayScalarFloat32(float32(1 - beta1))
	defer oneMinusB1.Free()
	b2 := bridge.NewArrayScalarFloat32(float32(beta2))
	defer b2.Free()
	oneMinusB2 := bridge.NewArrayScalarFloat32(float32(1 - beta2))
	defer oneMinusB2.Free()
	epsArr := bridge.NewArrayScalarFloat32(float32(eps))
	defer epsArr.Free()
	lrArr := bridge.NewArrayScalarFloat32(float32(lr))
	defer lrArr.Free()
	wdArr := bridge.NewArrayScalarFloat32(float32(wd))
	defer wdArr.Free()
	bc1Arr := bridge.NewArrayScalarFloat32(float32(biasCorr1))
	defer bc1Arr.Free()
	bc2Arr := bridge.NewArrayScalarFloat32(float32(biasCorr2))
	defer bc2Arr.Free()

	// m_new = beta1*m + (1-beta1)*g
	mNew := bridge.Add(bridge.Multiply(b1, mArr, s), bridge.Multiply(oneMinusB1, gArr, s), s)
	defer mNew.Free()

	// v_new = beta2*v + (1-beta2)*g*g
	gSq := bridge.Multiply(gArr, gArr, s)
	defer gSq.Free()
	vNew := bridge.Add(bridge.Multiply(b2, vArr, s), bridge.Multiply(oneMinusB2, gSq, s), s)
	defer vNew.Free()

	// mHat = m_new / biasCorr1
	mHat := bridge.Divide(mNew, bc1Arr, s)
	defer mHat.Free()

	// vHat = v_new / biasCorr2
	vHat := bridge.Divide(vNew, bc2Arr, s)
	defer vHat.Free()

	// update = mHat / (sqrt(vHat) + eps)
	sqrtV := bridge.Sqrt(vHat, s)
	defer sqrtV.Free()
	denom := bridge.Add(sqrtV, epsArr, s)
	defer denom.Free()
	update := bridge.Divide(mHat, denom, s)
	defer update.Free()

	var finalUpdate *bridge.Array
	if cautious {
		// mask = (update * grad > 0) ? 1.0 : 0.0
		prod := bridge.Multiply(update, gArr, s)
		defer prod.Free()
		zero := bridge.NewArrayScalarFloat32(0)
		defer zero.Free()
		mask := bridge.Greater(prod, zero, s)
		defer mask.Free()
		// Cast bool mask to float32
		maskF := bridge.AsType(mask, bridge.DTypeFloat32, s)
		defer maskF.Free()
		finalUpdate = bridge.Multiply(update, maskF, s)
		defer finalUpdate.Free()
	} else {
		finalUpdate = update
	}

	// w_new = w - lr*update - lr*wd*w
	lrUpdate := bridge.Multiply(lrArr, finalUpdate, s)
	defer lrUpdate.Free()
	lrWdW := bridge.Multiply(lrArr, bridge.Multiply(wdArr, wArr, s), s)
	defer lrWdW.Free()
	wNew := bridge.Subtract(bridge.Subtract(wArr, lrUpdate, s), lrWdW, s)
	defer wNew.Free()

	// Read back m, v, w
	readFloat32Into(mNew, m)
	readFloat32Into(vNew, v)
	readFloat32Into(wNew, w)

	runtime.KeepAlive(grad)
	runtime.KeepAlive(m)
	runtime.KeepAlive(v)
	runtime.KeepAlive(w)
}

// AdamWBatchUpdateFloat32 performs Cautious AdamW on ALL parameters at once.
// Each parameter group has its own LR but shares beta/eps/wd/biasCorr.
// All parameter slices are concatenated into one flat GPU array, processed in
// a single set of ~15 Metal kernel dispatches, then scattered back.
// This amortizes the kernel dispatch overhead across all parameters.
func AdamWBatchUpdateFloat32(
	grads, ms, vs, ws [][]float32,
	lrs []float64,
	beta1, beta2, eps, wd, biasCorr1, biasCorr2 float64,
	cautious bool,
) {
	if len(grads) == 0 {
		return
	}

	// Compute total size and build offset map.
	totalSize := 0
	offsets := make([]int, len(grads))
	for i, g := range grads {
		offsets[i] = totalSize
		totalSize += len(g)
	}

	// Concatenate all parameters into flat buffers.
	flatGrad := make([]float32, totalSize)
	flatM := make([]float32, totalSize)
	flatV := make([]float32, totalSize)
	flatW := make([]float32, totalSize)
	flatLR := make([]float32, totalSize)

	for i := range grads {
		off := offsets[i]
		n := len(grads[i])
		copy(flatGrad[off:off+n], grads[i])
		copy(flatM[off:off+n], ms[i])
		copy(flatV[off:off+n], vs[i])
		copy(flatW[off:off+n], ws[i])
		lr32 := float32(lrs[i])
		for j := off; j < off+n; j++ {
			flatLR[j] = lr32
		}
	}

	s := bridge.DefaultGPUStream()
	defer s.Free()

	gArr := arrayFromFloat32(flatGrad)
	defer gArr.Free()
	mArr := arrayFromFloat32(flatM)
	defer mArr.Free()
	vArr := arrayFromFloat32(flatV)
	defer vArr.Free()
	wArr := arrayFromFloat32(flatW)
	defer wArr.Free()
	lrArr := arrayFromFloat32(flatLR)
	defer lrArr.Free()

	b1 := bridge.NewArrayScalarFloat32(float32(beta1))
	defer b1.Free()
	oneMinusB1 := bridge.NewArrayScalarFloat32(float32(1 - beta1))
	defer oneMinusB1.Free()
	b2 := bridge.NewArrayScalarFloat32(float32(beta2))
	defer b2.Free()
	oneMinusB2 := bridge.NewArrayScalarFloat32(float32(1 - beta2))
	defer oneMinusB2.Free()
	epsArr := bridge.NewArrayScalarFloat32(float32(eps))
	defer epsArr.Free()
	wdArr := bridge.NewArrayScalarFloat32(float32(wd))
	defer wdArr.Free()
	bc1Arr := bridge.NewArrayScalarFloat32(float32(biasCorr1))
	defer bc1Arr.Free()
	bc2Arr := bridge.NewArrayScalarFloat32(float32(biasCorr2))
	defer bc2Arr.Free()

	// m_new = beta1*m + (1-beta1)*g
	mNew := bridge.Add(bridge.Multiply(b1, mArr, s), bridge.Multiply(oneMinusB1, gArr, s), s)
	defer mNew.Free()

	// v_new = beta2*v + (1-beta2)*g*g
	gSq := bridge.Multiply(gArr, gArr, s)
	defer gSq.Free()
	vNew := bridge.Add(bridge.Multiply(b2, vArr, s), bridge.Multiply(oneMinusB2, gSq, s), s)
	defer vNew.Free()

	// update = (m_new/biasCorr1) / (sqrt(v_new/biasCorr2) + eps)
	mHat := bridge.Divide(mNew, bc1Arr, s)
	defer mHat.Free()
	vHat := bridge.Divide(vNew, bc2Arr, s)
	defer vHat.Free()
	sqrtV := bridge.Sqrt(vHat, s)
	defer sqrtV.Free()
	denom := bridge.Add(sqrtV, epsArr, s)
	defer denom.Free()
	update := bridge.Divide(mHat, denom, s)
	defer update.Free()

	var finalUpdate *bridge.Array
	if cautious {
		prod := bridge.Multiply(update, gArr, s)
		defer prod.Free()
		zero := bridge.NewArrayScalarFloat32(0)
		defer zero.Free()
		mask := bridge.Greater(prod, zero, s)
		defer mask.Free()
		maskF := bridge.AsType(mask, bridge.DTypeFloat32, s)
		defer maskF.Free()
		finalUpdate = bridge.Multiply(update, maskF, s)
		defer finalUpdate.Free()
	} else {
		finalUpdate = update
	}

	// w_new = w - lr*update - lr*wd*w  (lr is per-element vector)
	lrUpdate := bridge.Multiply(lrArr, finalUpdate, s)
	defer lrUpdate.Free()
	lrWdW := bridge.Multiply(lrArr, bridge.Multiply(wdArr, wArr, s), s)
	defer lrWdW.Free()
	wNew := bridge.Subtract(bridge.Subtract(wArr, lrUpdate, s), lrWdW, s)
	defer wNew.Free()

	// Eval all three outputs at once (single Metal dispatch).
	if err := bridge.Eval(mNew, vNew, wNew); err != nil {
		panic("mlx batch optimizer: eval failed: " + err.Error())
	}

	// Read results back and scatter to individual parameter slices.
	mPtr := mNew.DataPtr()
	vPtr := vNew.DataPtr()
	wPtr := wNew.DataPtr()

	for i := range grads {
		off := offsets[i]
		n := len(grads[i])
		nbytes := n * 4
		copy(
			unsafe.Slice((*byte)(unsafe.Pointer(&ms[i][0])), nbytes),
			unsafe.Slice((*byte)(unsafe.Add(mPtr, uintptr(off*4))), nbytes),
		)
		copy(
			unsafe.Slice((*byte)(unsafe.Pointer(&vs[i][0])), nbytes),
			unsafe.Slice((*byte)(unsafe.Add(vPtr, uintptr(off*4))), nbytes),
		)
		copy(
			unsafe.Slice((*byte)(unsafe.Pointer(&ws[i][0])), nbytes),
			unsafe.Slice((*byte)(unsafe.Add(wPtr, uintptr(off*4))), nbytes),
		)
	}

	runtime.KeepAlive(grads)
	runtime.KeepAlive(ms)
	runtime.KeepAlive(vs)
	runtime.KeepAlive(ws)
}

// ScheduleFreeUpdateFloat32 performs Schedule-Free AdamW on the GPU via MLX.
// z, v, x are updated in-place. y receives the new parameter values.
//
// This is the GPU equivalent of optimizer_cpu.go:scheduleFreeUpdateVarInPlace.
func ScheduleFreeUpdateFloat32(gradData, z, v, x, y []float32,
	lr, beta1, beta2, eps, wd, biasCorr2, ck float64,
) {
	s := bridge.DefaultGPUStream()
	defer s.Free()

	gArr := arrayFromFloat32(gradData)
	defer gArr.Free()
	zArr := arrayFromFloat32(z)
	defer zArr.Free()
	vArr := arrayFromFloat32(v)
	defer vArr.Free()
	xArr := arrayFromFloat32(x)
	defer xArr.Free()

	b1 := bridge.NewArrayScalarFloat32(float32(beta1))
	defer b1.Free()
	oneMinusB1 := bridge.NewArrayScalarFloat32(float32(1 - beta1))
	defer oneMinusB1.Free()
	b2 := bridge.NewArrayScalarFloat32(float32(beta2))
	defer b2.Free()
	oneMinusB2 := bridge.NewArrayScalarFloat32(float32(1 - beta2))
	defer oneMinusB2.Free()
	epsArr := bridge.NewArrayScalarFloat32(float32(eps))
	defer epsArr.Free()
	lrArr := bridge.NewArrayScalarFloat32(float32(lr))
	defer lrArr.Free()
	wdArr := bridge.NewArrayScalarFloat32(float32(wd))
	defer wdArr.Free()
	bc2Arr := bridge.NewArrayScalarFloat32(float32(biasCorr2))
	defer bc2Arr.Free()
	ckArr := bridge.NewArrayScalarFloat32(float32(ck))
	defer ckArr.Free()
	oneMinusCk := bridge.NewArrayScalarFloat32(float32(1 - ck))
	defer oneMinusCk.Free()

	// y_eval = beta1*x + (1-beta1)*z
	yEval := bridge.Add(bridge.Multiply(b1, xArr, s), bridge.Multiply(oneMinusB1, zArr, s), s)
	defer yEval.Free()

	// z = z - lr*wd*y_eval
	zNew := bridge.Subtract(zArr, bridge.Multiply(lrArr, bridge.Multiply(wdArr, yEval, s), s), s)
	defer zNew.Free()

	// v_new = beta2*v + (1-beta2)*g*g
	gSq := bridge.Multiply(gArr, gArr, s)
	defer gSq.Free()
	vNew := bridge.Add(bridge.Multiply(b2, vArr, s), bridge.Multiply(oneMinusB2, gSq, s), s)
	defer vNew.Free()

	// vHat = v_new / biasCorr2
	vHat := bridge.Divide(vNew, bc2Arr, s)
	defer vHat.Free()

	// z = z - lr * g / (sqrt(vHat) + eps)
	sqrtV := bridge.Sqrt(vHat, s)
	defer sqrtV.Free()
	denominator := bridge.Add(sqrtV, epsArr, s)
	defer denominator.Free()
	gradStep := bridge.Divide(gArr, denominator, s)
	defer gradStep.Free()
	zFinal := bridge.Subtract(zNew, bridge.Multiply(lrArr, gradStep, s), s)
	defer zFinal.Free()

	// x_new = (1-ck)*x + ck*z
	xNew := bridge.Add(bridge.Multiply(oneMinusCk, xArr, s), bridge.Multiply(ckArr, zFinal, s), s)
	defer xNew.Free()

	// y_new = beta1*x_new + (1-beta1)*z_final
	yNew := bridge.Add(bridge.Multiply(b1, xNew, s), bridge.Multiply(oneMinusB1, zFinal, s), s)
	defer yNew.Free()

	// Read back z, v, x, y
	readFloat32Into(zFinal, z)
	readFloat32Into(vNew, v)
	readFloat32Into(xNew, x)
	readFloat32Into(yNew, y)

	runtime.KeepAlive(gradData)
	runtime.KeepAlive(z)
	runtime.KeepAlive(v)
	runtime.KeepAlive(x)
	runtime.KeepAlive(y)
}
