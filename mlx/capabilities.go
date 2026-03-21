// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mlx

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// OpCapabilityProfile refines the coarse supported/not-supported capability table
// with execution-path and training-safety metadata for MLX-specific routing.
type OpCapabilityProfile struct {
	Supported            bool
	CTapeSupported       bool
	GeneralCaseSupported bool
	TrainSafe            bool
	Notes                string
}

// backendCapabilities declares which ops and dtypes the MLX backend supports.
var backendCapabilities = backends.Capabilities{
	Functions: true,

	Operations: map[backends.OpType]bool{
		// Inputs
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Unary math
		backends.OpTypeAbs:        true,
		backends.OpTypeNeg:        true,
		backends.OpTypeSqrt:       true,
		backends.OpTypeRsqrt:      true,
		backends.OpTypeExp:        true,
		backends.OpTypeExpm1:      true,
		backends.OpTypeLog:        true,
		backends.OpTypeLog1p:      true,
		backends.OpTypeSin:        true,
		backends.OpTypeCos:        true,
		backends.OpTypeTanh:       true,
		backends.OpTypeLogistic:   true,
		backends.OpTypeErf:        true,
		backends.OpTypeFloor:      true,
		backends.OpTypeCeil:       true,
		backends.OpTypeRound:      true,
		backends.OpTypeSign:       true,
		backends.OpTypeLogicalNot: true,
		backends.OpTypeBitwiseNot: true,
		backends.OpTypeIsFinite:   true,
		backends.OpTypeIsNaN:      true,
		backends.OpTypeIdentity:   true,

		// Binary math
		backends.OpTypeAdd:   true,
		backends.OpTypeSub:   true,
		backends.OpTypeMul:   true,
		backends.OpTypeDiv:   true,
		backends.OpTypeRem:   true,
		backends.OpTypePow:   true,
		backends.OpTypeMax:   true,
		backends.OpTypeMin:   true,
		backends.OpTypeAtan2: true,

		// Logical
		backends.OpTypeLogicalAnd: true,
		backends.OpTypeLogicalOr:  true,
		backends.OpTypeLogicalXor: true,

		// Bitwise
		backends.OpTypeBitwiseAnd:           true,
		backends.OpTypeBitwiseOr:            true,
		backends.OpTypeBitwiseXor:           true,
		backends.OpTypeShiftLeft:            true,
		backends.OpTypeShiftRightArithmetic: true,
		backends.OpTypeShiftRightLogical:    true,

		// Comparison
		backends.OpTypeEqual:          true,
		backends.OpTypeNotEqual:       true,
		backends.OpTypeGreaterThan:    true,
		backends.OpTypeGreaterOrEqual: true,
		backends.OpTypeLessThan:       true,
		backends.OpTypeLessOrEqual:    true,

		// TotalOrder comparisons (treated same as regular)
		backends.OpTypeEqualTotalOrder:          true,
		backends.OpTypeNotEqualTotalOrder:       true,
		backends.OpTypeGreaterThanTotalOrder:    true,
		backends.OpTypeGreaterOrEqualTotalOrder: true,
		backends.OpTypeLessThanTotalOrder:       true,
		backends.OpTypeLessOrEqualTotalOrder:    true,

		// Shape operations
		backends.OpTypeReshape:        true,
		backends.OpTypeTranspose:      true,
		backends.OpTypeConvertDType:   true,
		backends.OpTypeBroadcastInDim: true,
		backends.OpTypeWhere:          true,
		backends.OpTypeClamp:          true,
		backends.OpTypeSlice:          true,
		backends.OpTypeConcatenate:    true,
		backends.OpTypeReverse:        true,
		backends.OpTypeIota:           true,
		backends.OpTypePad:            true,

		// Matrix operations
		backends.OpTypeDotGeneral: true,

		// Reductions
		backends.OpTypeReduceSum:     true,
		backends.OpTypeReduceMax:     true,
		backends.OpTypeReduceMin:     true,
		backends.OpTypeReduceProduct: true,

		// Logical reductions
		backends.OpTypeReduceLogicalAnd: true,
		backends.OpTypeReduceLogicalOr:  true,

		// ArgMin/ArgMax
		backends.OpTypeArgMinMax: true,

		// Batch normalization
		backends.OpTypeBatchNormForInference: true,
		backends.OpTypeBatchNormForTraining:  true,
		backends.OpTypeBatchNormGradient:     false, // computed via autograd

		// Gather/Scatter
		backends.OpTypeGather:     true,
		backends.OpTypeScatterSum: true,
		backends.OpTypeScatterMax: true,
		backends.OpTypeScatterMin: true,

		// DynamicSlice / DynamicUpdateSlice
		backends.OpTypeDynamicSlice:       true,
		backends.OpTypeDynamicUpdateSlice: true,

		// RNG
		backends.OpTypeRNGBitGenerator: true,

		// Convolution
		backends.OpTypeConvGeneral: true,

		// Pooling
		backends.OpTypeReduceWindow:        true,
		backends.OpTypeSelectAndScatterMax: false, // not yet implemented

		// Fused operations
		backends.OpTypeFusedSoftmax:                   true,
		backends.OpTypeFusedLayerNorm:                 true,
		backends.OpTypeFusedGelu:                      true,
		backends.OpTypeFusedDense:                     true,
		backends.OpTypeFusedScaledDotProductAttention: true,
		backends.OpTypeFusedAttentionQKVProjection:    true,

		// Control flow
		backends.OpTypeWhile: true,
		backends.OpTypeIf:    true,
		backends.OpTypeSort:  true,
		backends.OpTypeCall:  true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Float32:   true,
		dtypes.Float16:   true,
		dtypes.Float64:   true,
		dtypes.BFloat16:  true,
		dtypes.Bool:      true,
		dtypes.Int8:      true,
		dtypes.Int16:     true,
		dtypes.Int32:     true,
		dtypes.Int64:     true,
		dtypes.Uint8:     true,
		dtypes.Uint16:    true,
		dtypes.Uint32:    true,
		dtypes.Uint64:    true,
		dtypes.Complex64: true,
	},
}

// opCapabilityProfiles captures the operations whose practical behavior differs
// from the coarse backendCapabilities table. Any op not listed here defaults to:
// supported => ctape_supported => general_case_supported => train_safe.
var opCapabilityProfiles = map[backends.OpType]OpCapabilityProfile{
	backends.OpTypeBatchNormForInference: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Implemented via Go-callback compound op; functional but not recommended for large training graphs.",
	},
	backends.OpTypeBatchNormForTraining: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Implemented via Go-callback compound op; functional but not recommended for large training graphs.",
	},
	backends.OpTypeGather: {
		Supported:            true,
		CTapeSupported:       true,
		GeneralCaseSupported: false,
		TrainSafe:            true,
		Notes:                "Only simple gather forms are implemented; general case is not supported.",
	},
	backends.OpTypeScatterSum: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: false,
		TrainSafe:            false,
		Notes:                "Complex scatter forms fall back to Go replay; only restricted cases are suitable for training.",
	},
	backends.OpTypeScatterMax: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: false,
		TrainSafe:            false,
		Notes:                "Complex scatter forms fall back to Go replay; only restricted cases are suitable for training.",
	},
	backends.OpTypeScatterMin: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: false,
		TrainSafe:            false,
		Notes:                "Complex scatter forms fall back to Go replay; only restricted cases are suitable for training.",
	},
	backends.OpTypeDynamicSlice: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Dynamic slice currently forces raw Go-closure execution.",
	},
	backends.OpTypeDynamicUpdateSlice: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Dynamic update slice currently forces raw Go-closure execution.",
	},
	backends.OpTypeRNGBitGenerator: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "RNG path currently forces raw Go-closure execution.",
	},
	backends.OpTypeReduceWindow: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "ReduceWindow currently uses a compound Go-callback path.",
	},
	backends.OpTypeFusedAttentionQKVProjection: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Fused QKV projection currently uses a compound Go-callback path.",
	},
	backends.OpTypeWhile: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Control flow executes through the control-flow executor, not the steady-state C-tape path.",
	},
	backends.OpTypeIf: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Control flow executes through the control-flow executor, not the steady-state C-tape path.",
	},
	backends.OpTypeSort: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Control flow executes through the control-flow executor, not the steady-state C-tape path.",
	},
	backends.OpTypeCall: {
		Supported:            true,
		CTapeSupported:       false,
		GeneralCaseSupported: true,
		TrainSafe:            false,
		Notes:                "Control flow executes through the control-flow executor, not the steady-state C-tape path.",
	},
}

// CapabilityProfile returns a more precise MLX support profile for an operation.
// It preserves the existing coarse Capabilities contract while exposing MLX-
// specific execution caveats to callers that care about training safety.
func (b *Backend) CapabilityProfile(op backends.OpType) OpCapabilityProfile {
	if profile, found := opCapabilityProfiles[op]; found {
		return profile
	}
	supported := backendCapabilities.Operations[op]
	return OpCapabilityProfile{
		Supported:            supported,
		CTapeSupported:       supported,
		GeneralCaseSupported: supported,
		TrainSafe:            supported,
	}
}
