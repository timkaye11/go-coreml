// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package mpsgraph

import (
	"slices"

	"github.com/gomlx/go-coreml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Gather implements the XLA-style Gather operation.
//
// Strategy: decompose the general XLA Gather into MPSGraph operations.
//
// For the common embedding-lookup pattern (single indexed axis, gather full slices),
// we use MPSGraph's gatherAlongAxis. For the general case, we decompose into
// reshape + gatherND + reshape.
func (f *Function) Gather(
	operand, startIndices backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Value, error) {
	operandNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: operand")
	}
	indicesNode, err := f.resolveNode(startIndices)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: startIndices")
	}

	outShape, err := shapeinference.Gather(
		operandNode.shape, indicesNode.shape,
		indexVectorAxis, offsetOutputAxes, collapsedSliceAxes,
		startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, errors.Wrap(err, "Gather")
	}

	// Try fast path: simple embedding lookup along a single axis.
	if result, ok := f.gatherEmbeddingLookup(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape); ok {
		return result, nil
	}

	// General path: decompose XLA Gather into MPSGraph operations.
	return f.gatherGeneral(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape)
}

// gatherEmbeddingLookup handles the common pattern: gather full slices along one axis.
// Pattern: startIndexMap = [axis], sliceSizes[axis] = 1, collapsedSliceAxes = [axis],
// and all other sliceSizes are the full dimension.
func (f *Function) gatherEmbeddingLookup(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (backends.Value, bool) {
	// Must have exactly one axis in startIndexMap.
	if len(startIndexMap) != 1 {
		return nil, false
	}
	gatherAxis := startIndexMap[0]

	// sliceSizes[gatherAxis] must be 1.
	if sliceSizes[gatherAxis] != 1 {
		return nil, false
	}

	// gatherAxis must be in collapsedSliceAxes.
	if !slices.Contains(collapsedSliceAxes, gatherAxis) {
		return nil, false
	}

	// All other sliceSizes must be the full dimension.
	for i, sz := range sliceSizes {
		if i == gatherAxis {
			continue
		}
		if sz != operandNode.shape.Dimensions[i] {
			return nil, false
		}
	}

	// indexVectorAxis should be at the end (or startIndices.Rank() for implicit size-1 axis).
	idxRank := indicesNode.shape.Rank()
	if indexVectorAxis != idxRank-1 && indexVectorAxis != idxRank {
		return nil, false
	}

	// Remove the indexVector axis (size 1) from indices to get batch dims.
	idxTensor := indicesNode.tensor
	var err error

	// Compute batch dimensions (all index dims except indexVectorAxis).
	batchDims := make([]int64, 0, idxRank)
	for i, d := range indicesNode.shape.Dimensions {
		if i != indexVectorAxis || indexVectorAxis >= idxRank {
			batchDims = append(batchDims, int64(d))
		}
	}
	if len(batchDims) == 0 {
		batchDims = []int64{1} // Scalar index becomes [1].
	}

	if indexVectorAxis < idxRank {
		// Remove the indexVector axis (size 1) by reshaping.
		idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
		if err != nil {
			return nil, false
		}
	}

	// MPSGraph's gatherAlongAxis requires indices to have the same rank as operand.
	// We need to reshape indices to match operand rank, inserting size-1 for non-gather axes,
	// then broadcast to match the full gather output shape for gatherAlongAxis.
	operandRank := operandNode.shape.Rank()

	// Build the shape for indices matching operand rank: batch dims along gatherAxis,
	// size-1 for all other axes.
	idxFullShape := make([]int64, operandRank)
	for i := range operandRank {
		if i == gatherAxis {
			// Flatten all batch dims into this axis.
			batchSize := int64(1)
			for _, d := range batchDims {
				batchSize *= d
			}
			idxFullShape[i] = batchSize
		} else {
			idxFullShape[i] = 1
		}
	}
	idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
	if err != nil {
		return nil, false
	}

	// Broadcast indices to match the gatherAlongAxis output shape:
	// same as operand but with gatherAxis replaced by the batch size.
	broadcastShape := make([]int64, operandRank)
	for i, d := range operandNode.shape.Dimensions {
		if i == gatherAxis {
			broadcastShape[i] = idxFullShape[i]
		} else {
			broadcastShape[i] = int64(d)
		}
	}
	idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
	if err != nil {
		return nil, false
	}

	// Use gatherAlongAxis — indices and operand now have the same rank.
	result, err := f.ctx().GatherAlongAxis(operandNode.tensor, idxTensor, gatherAxis)
	if err != nil {
		return nil, false
	}

	// After gatherAlongAxis, the result has the gathered dimension at gatherAxis.
	// For the output, batch dims should come first. If gatherAxis > 0, we need
	// to transpose the gatherAxis to position 0, then reshape to output shape.
	if gatherAxis > 0 {
		// Build permutation: move gatherAxis to front, keep rest in order.
		perm := make([]int, operandRank)
		perm[0] = gatherAxis
		idx := 1
		for i := range operandRank {
			if i != gatherAxis {
				perm[idx] = i
				idx++
			}
		}
		result, err = f.ctx().Transpose(result, perm)
		if err != nil {
			return nil, false
		}
	}

	// Reshape to the expected output shape.
	outDims := make([]int64, outShape.Rank())
	for i, d := range outShape.Dimensions {
		outDims[i] = int64(d)
	}
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, false
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, true
}

// gatherGeneral implements the full XLA Gather semantics by decomposing into
// flatten indices → gatherND → reshape output.
func (f *Function) gatherGeneral(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (backends.Value, error) {
	operandShape := operandNode.shape
	indicesShape := indicesNode.shape
	operandRank := operandShape.Rank()

	// Step 1: Extract the index vectors from startIndices.
	// The indexVectorAxis contains the multi-dimensional index.
	idxTensor := indicesNode.tensor
	var err error

	// If indexVectorAxis == indicesShape.Rank(), there's an implicit axis of size 1.
	if indexVectorAxis == indicesShape.Rank() {
		// Add a trailing axis of size 1.
		newDims := make([]int64, indicesShape.Rank()+1)
		for i, d := range indicesShape.Dimensions {
			newDims[i] = int64(d)
		}
		newDims[indicesShape.Rank()] = 1
		idxTensor, err = f.ctx().Reshape(idxTensor, newDims)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: reshape indices for implicit axis")
		}
	}

	// Step 2: Determine batch dimensions (all dims except indexVectorAxis).
	// Flatten batch dims into one.
	batchSize := 1
	for i := range indicesShape.Rank() {
		if i != indexVectorAxis || indexVectorAxis == indicesShape.Rank() {
			if i < len(indicesShape.Dimensions) {
				batchSize *= indicesShape.Dimensions[i]
			}
		}
	}

	indexVectorSize := len(startIndexMap)

	// Step 3: If startIndexMap is a contiguous prefix [0,1,...,k-1], we can use gatherND directly.
	// Otherwise, we need to remap indices.
	isContiguousPrefix := true
	for i, v := range startIndexMap {
		if v != i {
			isContiguousPrefix = false
			break
		}
	}

	// Step 4: Check if all slice sizes beyond the indexed axes are the full dimension
	// and all indexed axes have slice size 1 (with collapsed).
	allSlicesFullOrCollapsed := true
	collapsedSet := make(map[int]bool)
	for _, a := range collapsedSliceAxes {
		collapsedSet[a] = true
	}
	indexedSet := make(map[int]bool)
	for _, a := range startIndexMap {
		indexedSet[a] = true
	}
	for i := range operandRank {
		if collapsedSet[i] {
			if sliceSizes[i] != 1 {
				allSlicesFullOrCollapsed = false
				break
			}
		} else if !indexedSet[i] {
			if sliceSizes[i] != operandShape.Dimensions[i] {
				allSlicesFullOrCollapsed = false
				break
			}
		}
	}

	if isContiguousPrefix && allSlicesFullOrCollapsed && len(collapsedSliceAxes) == len(startIndexMap) {
		// Optimal path: gatherND with contiguous index prefix.
		// Reshape indices to [batchSize, indexVectorSize] for gatherND.
		idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
		if err != nil {
			return nil, errors.Wrap(err, "Gather: reshape indices for gatherND")
		}

		result, err := f.ctx().GatherND(operandNode.tensor, idxTensor, 0)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: gatherND")
		}

		// Reshape to output shape.
		outDims := make([]int64, outShape.Rank())
		for i, d := range outShape.Dimensions {
			outDims[i] = int64(d)
		}
		result, err = f.ctx().Reshape(result, outDims)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: reshape output")
		}
		return &graphNode{tensor: result, shape: outShape, owner: f}, nil
	}

	// Fallback: decompose into per-element gather using gatherND with index remapping.
	// This handles the most general case by building the full index tensor.

	// Build remapped indices: for each batch element, create the full operand index.
	// The startIndexMap tells us which operand axes are indexed by each position in the index vector.
	// Non-indexed axes start at 0 with full slice.

	// For non-contiguous startIndexMap, we need to rearrange the index columns.
	// Reshape indices to [batchSize, indexVectorSize].
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape indices")
	}

	// If startIndexMap is not [0, 1, ..., k-1], we need to build a full-rank index tensor
	// by inserting zeros for non-indexed axes and rearranging.
	// For now, handle the case where startIndexMap maps to a contiguous set of axes
	// (even if not starting at 0) by transposing the operand first.

	// General strategy: transpose operand so that startIndexMap axes come first,
	// then use gatherND on the transposed operand.
	perm := make([]int, operandRank)
	copy(perm, startIndexMap)
	idx := len(startIndexMap)
	for i := range operandRank {
		if !indexedSet[i] {
			perm[idx] = i
			idx++
		}
	}

	operandTensor := operandNode.tensor
	needsTranspose := false
	for i, v := range perm {
		if v != i {
			needsTranspose = true
			break
		}
	}
	if needsTranspose {
		operandTensor, err = f.ctx().Transpose(operandTensor, perm)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: transpose operand")
		}
	}

	// Now the first len(startIndexMap) axes of operand are the indexed ones.
	// Use gatherND with the indices.
	result, err := f.ctx().GatherND(operandTensor, idxTensor, 0)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: gatherND general")
	}

	// Reshape to output shape.
	outDims := make([]int64, outShape.Rank())
	for i, d := range outShape.Dimensions {
		outDims[i] = int64(d)
	}
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape output general")
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

// ===========================================================================
// Scatter Operations
// ===========================================================================

// scatterImpl is the common implementation for ScatterSum, ScatterMax, ScatterMin.
func (f *Function) scatterImpl(
	opName string,
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
	scatterMode int,
) (backends.Value, error) {
	operandNode, err := f.resolveNode(operandOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": operand")
	}
	indicesNode, err := f.resolveNode(scatterIndicesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": indices")
	}
	updatesNode, err := f.resolveNode(updatesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": updates")
	}

	outShape, err := shapeinference.ScatterOp(
		operandNode.shape, indicesNode.shape, updatesNode.shape,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	// Try simple scatter along a single axis (common case for embedding gradients).
	if len(scatterAxesToOperandAxes) == 1 && len(insertedWindowAxes) == 1 {
		axis := scatterAxesToOperandAxes[0]

		// Flatten indices to remove indexVectorAxis, yielding batch dimensions.
		idxTensor := indicesNode.tensor
		idxRank := indicesNode.shape.Rank()
		batchDims := make([]int64, 0, idxRank)
		for i, d := range indicesNode.shape.Dimensions {
			if i != indexVectorAxis || indexVectorAxis >= idxRank {
				batchDims = append(batchDims, int64(d))
			}
		}
		if len(batchDims) == 0 {
			batchDims = []int64{1}
		}
		if indexVectorAxis < idxRank {
			idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
			if err != nil {
				return nil, errors.Wrap(err, opName+": reshape indices")
			}
		}

		// MPSGraph's scatterAlongAxis requires indices rank == updates rank.
		// Reshape indices to match updates rank: batch dims go into the scatter axis,
		// size-1 for all other axes, then broadcast to updates shape.
		updatesRank := updatesNode.shape.Rank()
		batchSize := int64(1)
		for _, d := range batchDims {
			batchSize *= d
		}
		idxFullShape := make([]int64, updatesRank)
		for i := range updatesRank {
			if i == axis {
				idxFullShape[i] = batchSize
			} else {
				idxFullShape[i] = 1
			}
		}
		idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape indices to updates rank")
		}

		// Broadcast to match updates shape.
		broadcastShape := make([]int64, updatesRank)
		for i, d := range updatesNode.shape.Dimensions {
			if i == axis {
				broadcastShape[i] = batchSize
			} else {
				broadcastShape[i] = int64(d)
			}
		}
		idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": broadcast indices")
		}

		result, err := f.ctx().ScatterAlongAxis(operandNode.tensor, idxTensor, updatesNode.tensor, axis, scatterMode)
		if err != nil {
			return nil, errors.Wrap(err, opName+": scatter along axis")
		}
		return &graphNode{tensor: result, shape: outShape, owner: f}, nil
	}

	// General case: use scatterND.
	// Reshape indices for scatterND.
	batchSize := 1
	for i, d := range indicesNode.shape.Dimensions {
		if i != indexVectorAxis {
			batchSize *= d
		}
	}
	indexVectorSize := len(scatterAxesToOperandAxes)

	idxTensor := indicesNode.tensor
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, opName+": reshape indices for scatterND")
	}

	// For scatterND, determine the output shape.
	outDims := make([]int64, outShape.Rank())
	for i, d := range outShape.Dimensions {
		outDims[i] = int64(d)
	}

	result, err := f.ctx().ScatterND(operandNode.tensor, idxTensor, updatesNode.tensor, outDims, scatterMode)
	if err != nil {
		return nil, errors.Wrap(err, opName+": scatterND")
	}
	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

func (f *Function) ScatterSum(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterSum", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeAdd)
}

func (f *Function) ScatterMax(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterMax", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMax)
}

func (f *Function) ScatterMin(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterMin", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMin)
}
