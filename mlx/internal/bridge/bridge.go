// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package bridge provides CGo bindings to Apple's MLX framework via mlx-c
// for GPU-accelerated tensor computation on Apple Silicon.
package bridge

/*
#cgo darwin CFLAGS: -I${SRCDIR}/deps/include
#cgo darwin LDFLAGS: -L${SRCDIR}/deps/lib -lmlxc -lmlx -lc++ -framework Metal -framework Foundation -framework Accelerate

#include "mlx/c/mlx.h"
#include "mlx/c/compile.h"
#include <stdlib.h>
#include <string.h>

// Wrapper to avoid CGo issues with __fp16 type.
static const void* mlx_array_data_float16_ptr(mlx_array arr) {
    return (const void*)mlx_array_data_float16(arr);
}

// Helper: check if mlx_array has a valid context.
static int mlx_array_ctx_is_null(mlx_array arr) {
    return arr.ctx == NULL;
}
static int mlx_stream_ctx_is_null(mlx_stream s) {
    return s.ctx == NULL;
}
static int mlx_vector_array_ctx_is_null(mlx_vector_array v) {
    return v.ctx == NULL;
}

// CGo trampoline: C function that forwards to Go callback via payload.
extern int goClosureCallback(mlx_vector_array* res, const mlx_vector_array inputs, void* payload);

static mlx_closure create_go_closure(void* payload) {
    return mlx_closure_new_func_payload(goClosureCallback, payload, NULL);
}

// =========================================================================
// C Tape Interpreter — replays a serialized instruction stream in pure C,
// eliminating CGo boundary crossings during steady-state execution.
// =========================================================================

// Opcodes for the tape interpreter.
enum tape_opcode {
    // Unary
    OP_ABS=0, OP_NEG, OP_SQRT, OP_RSQRT, OP_EXP, OP_EXPM1, OP_LOG, OP_LOG1P,
    OP_SIN, OP_COS, OP_TANH, OP_SIGMOID, OP_ERF, OP_FLOOR, OP_CEIL, OP_ROUND,
    OP_SIGN, OP_LOGICAL_NOT, OP_BITWISE_NOT, OP_ISNAN, OP_ISINF,
    // Binary
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REM, OP_POW, OP_MAX, OP_MIN,
    OP_ATAN2, OP_LOGICAL_AND, OP_LOGICAL_OR,
    OP_BITWISE_AND, OP_BITWISE_OR, OP_BITWISE_XOR, OP_LEFT_SHIFT, OP_RIGHT_SHIFT,
    OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
    OP_MATMUL,
    // Ternary
    OP_WHERE, OP_CLIP,
    // Shape
    OP_RESHAPE, OP_TRANSPOSE, OP_BROADCAST_TO, OP_SLICE, OP_SQUEEZE, OP_EXPAND_DIMS, OP_FLIP,
    // Reduce
    OP_REDUCE_SUM, OP_REDUCE_MAX, OP_REDUCE_MIN, OP_REDUCE_PROD, OP_REDUCE_ALL, OP_REDUCE_ANY,
    OP_ARGMIN, OP_ARGMAX,
    // Type
    OP_ASTYPE,
    // Special
    OP_SOFTMAX, OP_PAD, OP_CONCAT, OP_TAKE, OP_TAKE_ALONG_AXIS,
    OP_CONV1D, OP_CONV2D,
    OP_SCATTER_ADD, OP_SCATTER_MAX, OP_SCATTER_MIN,
    OP_FAST_LAYER_NORM, OP_FAST_RMS_NORM, OP_FAST_SDPA, OP_FAST_ROPE,
    OP_SLICE_UPDATE,
    // Scalar creation
    OP_NEW_SCALAR_F32, OP_NEW_SCALAR_I32, OP_NEW_SCALAR_BOOL,
    // Temp management
    OP_FREE_TEMP,
    // Stop gradient / copy
    OP_STOP_GRADIENT, OP_COPY,
    OP_COUNT // sentinel
};

// Payload for the C tape closure.
typedef struct {
    int32_t* instrs;          // flat instruction stream
    int32_t  instrs_len;      // total int32s in instrs
    int32_t  num_slots;       // total array slots
    int32_t* output_slots;    // which slots are outputs
    int32_t  num_outputs;
    mlx_array* const_arrays;  // pre-set constant arrays (borrowed refs)
    int32_t* const_slots;     // which slots hold constants
    int32_t  num_consts;
    int32_t* param_slots;     // which slots are parameters (filled from inputs)
    int32_t  num_params;
} tape_payload;

static void tape_payload_free(void* ptr) {
    tape_payload* p = (tape_payload*)ptr;
    if (p->instrs) free(p->instrs);
    if (p->output_slots) free(p->output_slots);
    if (p->const_arrays) {
        for (int32_t i = 0; i < p->num_consts; i++) {
            mlx_array_free(p->const_arrays[i]);
        }
        free(p->const_arrays);
    }
    if (p->const_slots) free(p->const_slots);
    if (p->param_slots) free(p->param_slots);
    free(p);
}

// Helper to read N int32s from the instruction stream and advance the pointer.
static inline const int32_t* read_ints(const int32_t* ip, int n, const int32_t** out) {
    *out = ip;
    return ip + n;
}

// The C tape interpreter. Called by MLX's compiled closure mechanism.
static int c_replay_tape(mlx_vector_array* res, const mlx_vector_array inputs, void* payload) {
    tape_payload* p = (tape_payload*)payload;
    mlx_stream stream = mlx_default_gpu_stream_new();

    // Allocate slot array.
    mlx_array* slots = (mlx_array*)calloc(p->num_slots, sizeof(mlx_array));
    if (!slots) return 1;

    // Fill parameter slots from inputs.
    size_t num_inputs = mlx_vector_array_size(inputs);
    for (int32_t i = 0; i < p->num_params && i < (int32_t)num_inputs; i++) {
        mlx_vector_array_get(&slots[p->param_slots[i]], inputs, i);
    }

    // Fill constant slots (retain references).
    for (int32_t i = 0; i < p->num_consts; i++) {
        mlx_array_set(&slots[p->const_slots[i]], p->const_arrays[i]);
    }

    // Interpret instruction stream.
    const int32_t* ip = p->instrs;
    const int32_t* end = p->instrs + p->instrs_len;

    while (ip < end) {
        int32_t opcode = *ip++;
        int32_t out_slot = *ip++;
        int32_t num_in = *ip++;
        // Read input slots
        const int32_t* in_slots;
        ip = read_ints(ip, num_in, &in_slots);
        int32_t num_p = *ip++;
        // Read params
        const int32_t* params;
        ip = read_ints(ip, num_p, &params);

        mlx_array result = {0};

        switch (opcode) {
        // ---- Unary ops ----
        case OP_ABS: mlx_abs(&result, slots[in_slots[0]], stream); break;
        case OP_NEG: mlx_negative(&result, slots[in_slots[0]], stream); break;
        case OP_SQRT: mlx_sqrt(&result, slots[in_slots[0]], stream); break;
        case OP_RSQRT: mlx_rsqrt(&result, slots[in_slots[0]], stream); break;
        case OP_EXP: mlx_exp(&result, slots[in_slots[0]], stream); break;
        case OP_EXPM1: mlx_expm1(&result, slots[in_slots[0]], stream); break;
        case OP_LOG: mlx_log(&result, slots[in_slots[0]], stream); break;
        case OP_LOG1P: mlx_log1p(&result, slots[in_slots[0]], stream); break;
        case OP_SIN: mlx_sin(&result, slots[in_slots[0]], stream); break;
        case OP_COS: mlx_cos(&result, slots[in_slots[0]], stream); break;
        case OP_TANH: mlx_tanh(&result, slots[in_slots[0]], stream); break;
        case OP_SIGMOID: mlx_sigmoid(&result, slots[in_slots[0]], stream); break;
        case OP_ERF: mlx_erf(&result, slots[in_slots[0]], stream); break;
        case OP_FLOOR: mlx_floor(&result, slots[in_slots[0]], stream); break;
        case OP_CEIL: mlx_ceil(&result, slots[in_slots[0]], stream); break;
        case OP_ROUND: mlx_round(&result, slots[in_slots[0]], 0, stream); break;
        case OP_SIGN: mlx_sign(&result, slots[in_slots[0]], stream); break;
        case OP_LOGICAL_NOT: mlx_logical_not(&result, slots[in_slots[0]], stream); break;
        case OP_BITWISE_NOT: mlx_bitwise_invert(&result, slots[in_slots[0]], stream); break;
        case OP_ISNAN: mlx_isnan(&result, slots[in_slots[0]], stream); break;
        case OP_ISINF: mlx_isinf(&result, slots[in_slots[0]], stream); break;

        // ---- Binary ops ----
        case OP_ADD: mlx_add(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_SUB: mlx_subtract(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_MUL: mlx_multiply(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_DIV: mlx_divide(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_REM: mlx_remainder(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_POW: mlx_power(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_MAX: mlx_maximum(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_MIN: mlx_minimum(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_ATAN2: mlx_arctan2(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_LOGICAL_AND: mlx_logical_and(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_LOGICAL_OR: mlx_logical_or(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_BITWISE_AND: mlx_bitwise_and(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_BITWISE_OR: mlx_bitwise_or(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_BITWISE_XOR: mlx_bitwise_xor(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_LEFT_SHIFT: mlx_left_shift(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_RIGHT_SHIFT: mlx_right_shift(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_EQ: mlx_equal(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_NE: mlx_not_equal(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_LT: mlx_less(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_LE: mlx_less_equal(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_GT: mlx_greater(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_GE: mlx_greater_equal(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;
        case OP_MATMUL: mlx_matmul(&result, slots[in_slots[0]], slots[in_slots[1]], stream); break;

        // ---- Ternary ops ----
        case OP_WHERE: mlx_where(&result, slots[in_slots[0]], slots[in_slots[1]], slots[in_slots[2]], stream); break;
        case OP_CLIP: mlx_clip(&result, slots[in_slots[0]], slots[in_slots[1]], slots[in_slots[2]], stream); break;

        // ---- Shape ops (params encode shapes/axes as [ndim, vals...]) ----
        case OP_RESHAPE: {
            int32_t ndim = params[0];
            int shape[ndim];
            for (int i = 0; i < ndim; i++) shape[i] = params[1+i];
            mlx_reshape(&result, slots[in_slots[0]], shape, ndim, stream);
            break;
        }
        case OP_TRANSPOSE: {
            int32_t naxes = params[0];
            if (naxes == 0) {
                mlx_transpose(&result, slots[in_slots[0]], stream);
            } else {
                int axes[naxes];
                for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
                mlx_transpose_axes(&result, slots[in_slots[0]], axes, naxes, stream);
            }
            break;
        }
        case OP_BROADCAST_TO: {
            int32_t ndim = params[0];
            int shape[ndim];
            for (int i = 0; i < ndim; i++) shape[i] = params[1+i];
            mlx_broadcast_to(&result, slots[in_slots[0]], shape, ndim, stream);
            break;
        }
        case OP_SLICE: {
            // params: [ndim, starts..., stops..., strides...]
            int32_t ndim = params[0];
            int starts[ndim], stops[ndim], strides[ndim];
            for (int i = 0; i < ndim; i++) {
                starts[i] = params[1 + i];
                stops[i] = params[1 + ndim + i];
                strides[i] = params[1 + 2*ndim + i];
            }
            mlx_slice(&result, slots[in_slots[0]], starts, ndim, stops, ndim, strides, ndim, stream);
            break;
        }
        case OP_SQUEEZE: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            mlx_squeeze_axes(&result, slots[in_slots[0]], axes, naxes, stream);
            break;
        }
        case OP_EXPAND_DIMS: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            mlx_expand_dims_axes(&result, slots[in_slots[0]], axes, naxes, stream);
            break;
        }
        case OP_FLIP: {
            // Flip is implemented as take with reversed indices per axis.
            // params: [naxes, axes...]
            int32_t naxes = params[0];
            // Retain input reference as starting point.
            mlx_array_set(&result, slots[in_slots[0]]);
            for (int i = 0; i < naxes; i++) {
                int axis = params[1+i];
                const int* shape = mlx_array_shape(result);
                int n = shape[axis];
                mlx_array indices = {0};
                mlx_arange(&indices, (double)(n-1), -1.0, -1.0, MLX_INT32, stream);
                mlx_array prev = result;
                result = (mlx_array){0};
                mlx_take_axis(&result, prev, indices, axis, stream);
                mlx_array_free(prev);
                mlx_array_free(indices);
            }
            break;
        }

        // ---- Reduce ops ----
        // params: [naxes, axes..., keepdims]
        case OP_REDUCE_SUM: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_sum_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_REDUCE_MAX: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_max_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_REDUCE_MIN: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_min_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_REDUCE_PROD: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_prod_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_REDUCE_ALL: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_all_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_REDUCE_ANY: {
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            bool kd = params[1+naxes];
            mlx_any_axes(&result, slots[in_slots[0]], axes, naxes, kd, stream);
            break;
        }
        case OP_ARGMIN: {
            // params: [axis, keepdims, out_dtype]
            int axis = params[0];
            bool kd = params[1];
            mlx_argmin_axis(&result, slots[in_slots[0]], axis, kd, stream);
            if (params[2] >= 0) {
                mlx_array casted;
                mlx_astype(&casted, result, (mlx_dtype)params[2], stream);
                mlx_array_free(result);
                result = casted;
            }
            break;
        }
        case OP_ARGMAX: {
            int axis = params[0];
            bool kd = params[1];
            mlx_argmax_axis(&result, slots[in_slots[0]], axis, kd, stream);
            if (params[2] >= 0) {
                mlx_array casted;
                mlx_astype(&casted, result, (mlx_dtype)params[2], stream);
                mlx_array_free(result);
                result = casted;
            }
            break;
        }

        // ---- Type ----
        case OP_ASTYPE: {
            mlx_astype(&result, slots[in_slots[0]], (mlx_dtype)params[0], stream);
            break;
        }

        // ---- Special ops ----
        case OP_SOFTMAX: {
            // params: [axis, precise]
            mlx_softmax_axis(&result, slots[in_slots[0]], params[0], (bool)params[1], stream);
            break;
        }
        case OP_PAD: {
            // params: [naxes, axes..., low_pads..., high_pads...]
            int32_t naxes = params[0];
            int axes[naxes], low[naxes], high[naxes];
            for (int i = 0; i < naxes; i++) {
                axes[i] = params[1+i];
                low[i] = params[1+naxes+i];
                high[i] = params[1+2*naxes+i];
            }
            char* mode = "constant";
            mlx_pad(&result, slots[in_slots[0]], axes, naxes, low, naxes, high, naxes, slots[in_slots[1]], mode, stream);
            break;
        }
        case OP_CONCAT: {
            // params: [axis]
            // inputs: [n arrays]
            mlx_array arrs[num_in];
            for (int i = 0; i < num_in; i++) arrs[i] = slots[in_slots[i]];
            mlx_vector_array vec = mlx_vector_array_new_data(arrs, num_in);
            mlx_concatenate_axis(&result, vec, params[0], stream);
            mlx_vector_array_free(vec);
            break;
        }
        case OP_TAKE: {
            // params: [axis]
            mlx_take_axis(&result, slots[in_slots[0]], slots[in_slots[1]], params[0], stream);
            break;
        }
        case OP_TAKE_ALONG_AXIS: {
            mlx_take_along_axis(&result, slots[in_slots[0]], slots[in_slots[1]], params[0], stream);
            break;
        }
        case OP_CONV1D: {
            // params: [stride, padding, dilation, groups]
            mlx_conv1d(&result, slots[in_slots[0]], slots[in_slots[1]],
                params[0], params[1], params[2], params[3], stream);
            break;
        }
        case OP_CONV2D: {
            // params: [stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups]
            mlx_conv2d(&result, slots[in_slots[0]], slots[in_slots[1]],
                params[0], params[1], params[2], params[3], params[4], params[5], params[6], stream);
            break;
        }
        case OP_SCATTER_ADD:
        case OP_SCATTER_MAX:
        case OP_SCATTER_MIN: {
            // inputs: [operand, indices, updates], params: [naxes, axes...]
            int32_t naxes = params[0];
            int axes[naxes];
            for (int i = 0; i < naxes; i++) axes[i] = params[1+i];
            mlx_array idx_arr = slots[in_slots[1]];
            mlx_vector_array idx_vec = mlx_vector_array_new_data(&idx_arr, 1);
            if (opcode == OP_SCATTER_ADD)
                mlx_scatter_add(&result, slots[in_slots[0]], idx_vec, slots[in_slots[2]], axes, naxes, stream);
            else if (opcode == OP_SCATTER_MAX)
                mlx_scatter_max(&result, slots[in_slots[0]], idx_vec, slots[in_slots[2]], axes, naxes, stream);
            else
                mlx_scatter_min(&result, slots[in_slots[0]], idx_vec, slots[in_slots[2]], axes, naxes, stream);
            mlx_vector_array_free(idx_vec);
            break;
        }
        case OP_FAST_LAYER_NORM: {
            // inputs: [x, weight_or_null, bias_or_null], params: [eps_bits, has_weight, has_bias]
            float eps;
            memcpy(&eps, &params[0], sizeof(float));
            mlx_array w = {0}, b = {0};
            if (params[1]) w = slots[in_slots[1]];
            if (params[2]) b = slots[in_slots[2]];
            mlx_fast_layer_norm(&result, slots[in_slots[0]], w, b, eps, stream);
            break;
        }
        case OP_FAST_RMS_NORM: {
            // inputs: [x, weight_or_null], params: [eps_bits, has_weight]
            float eps;
            memcpy(&eps, &params[0], sizeof(float));
            mlx_array w = {0};
            if (params[1]) w = slots[in_slots[1]];
            mlx_fast_rms_norm(&result, slots[in_slots[0]], w, eps, stream);
            break;
        }
        case OP_FAST_SDPA: {
            // inputs: [q, k, v, mask_or_null], params: [scale_bits, has_mask]
            float scale;
            memcpy(&scale, &params[0], sizeof(float));
            mlx_array mask_arr = {0};
            char* mask_mode = "none";
            if (params[1]) {
                mask_arr = slots[in_slots[3]];
                mask_mode = "additive";
            }
            mlx_array sinks = {0};
            mlx_fast_scaled_dot_product_attention(&result,
                slots[in_slots[0]], slots[in_slots[1]], slots[in_slots[2]],
                scale, mask_mode, mask_arr, sinks, stream);
            break;
        }
        case OP_FAST_ROPE: {
            // inputs: [x], params: [dims, traditional, base_bits, scale_bits, offset]
            int dims = params[0];
            bool traditional = params[1];
            float base, scale;
            memcpy(&base, &params[2], sizeof(float));
            memcpy(&scale, &params[3], sizeof(float));
            int offset = params[4];
            mlx_optional_float opt_base = {base, true};
            mlx_array freqs = {0};
            mlx_fast_rope(&result, slots[in_slots[0]], dims, traditional,
                opt_base, scale, offset, freqs, stream);
            break;
        }
        case OP_SLICE_UPDATE: {
            // params: [ndim, starts..., stops..., strides...]
            int32_t ndim = params[0];
            int starts[ndim], stops[ndim], strides[ndim];
            for (int i = 0; i < ndim; i++) {
                starts[i] = params[1 + i];
                stops[i] = params[1 + ndim + i];
                strides[i] = params[1 + 2*ndim + i];
            }
            mlx_slice_update(&result, slots[in_slots[0]], slots[in_slots[1]], starts, ndim, stops, ndim, strides, ndim, stream);
            break;
        }

        // ---- Scalar creation ----
        case OP_NEW_SCALAR_F32: {
            float val;
            memcpy(&val, &params[0], sizeof(float));
            result = mlx_array_new_float(val);
            break;
        }
        case OP_NEW_SCALAR_I32: {
            result = mlx_array_new_int(params[0]);
            break;
        }
        case OP_NEW_SCALAR_BOOL: {
            result = mlx_array_new_bool((bool)params[0]);
            break;
        }

        // ---- Temp management ----
        case OP_FREE_TEMP: {
            // out_slot is unused, in_slots are the temp slots to free
            for (int i = 0; i < num_in; i++) {
                if (slots[in_slots[i]].ctx != NULL) {
                    mlx_array_free(slots[in_slots[i]]);
                    slots[in_slots[i]].ctx = NULL;
                }
            }
            continue; // no result to store
        }

        case OP_STOP_GRADIENT: {
            mlx_stop_gradient(&result, slots[in_slots[0]], stream);
            break;
        }
        case OP_COPY: {
            mlx_copy(&result, slots[in_slots[0]], stream);
            break;
        }

        default:
            // Unknown opcode — clean up and fail.
            for (int32_t i = 0; i < p->num_slots; i++) {
                if (slots[i].ctx != NULL) mlx_array_free(slots[i]);
            }
            free(slots);
            mlx_stream_free(stream);
            return 1;
        }

        slots[out_slot] = result;
    }

    // Build output vector.
    mlx_array* out_arrs = (mlx_array*)malloc(p->num_outputs * sizeof(mlx_array));
    for (int32_t i = 0; i < p->num_outputs; i++) {
        out_arrs[i] = slots[p->output_slots[i]];
    }
    *res = mlx_vector_array_new_data(out_arrs, p->num_outputs);
    free(out_arrs);

    // Free non-output, non-param, non-const slots.
    // Actually, MLX manages refcounts. The output arrays are in the result vector.
    // We just free our slot copies for non-outputs.
    // For params: they were gotten from the input vector, so they have a refcount bump.
    // We should free all slots that are NOT outputs.
    for (int32_t i = 0; i < p->num_slots; i++) {
        if (slots[i].ctx == NULL) continue;
        bool is_output = false;
        for (int32_t j = 0; j < p->num_outputs; j++) {
            if (p->output_slots[j] == i) { is_output = true; break; }
        }
        if (!is_output) {
            mlx_array_free(slots[i]);
        }
    }

    free(slots);
    mlx_stream_free(stream);
    return 0;
}

static mlx_closure create_tape_closure(tape_payload* p) {
    return mlx_closure_new_func_payload(c_replay_tape, p, tape_payload_free);
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// DType mirrors MLX dtype enum.
type DType = C.mlx_dtype

var (
	DTypeBool      DType = C.MLX_BOOL
	DTypeUint8     DType = C.MLX_UINT8
	DTypeUint16    DType = C.MLX_UINT16
	DTypeUint32    DType = C.MLX_UINT32
	DTypeUint64    DType = C.MLX_UINT64
	DTypeInt8      DType = C.MLX_INT8
	DTypeInt16     DType = C.MLX_INT16
	DTypeInt32     DType = C.MLX_INT32
	DTypeInt64     DType = C.MLX_INT64
	DTypeFloat16   DType = C.MLX_FLOAT16
	DTypeBFloat16  DType = C.MLX_BFLOAT16
	DTypeFloat32   DType = C.MLX_FLOAT32
	DTypeFloat64   DType = C.MLX_FLOAT64
	DTypeComplex64 DType = C.MLX_COMPLEX64
)

// ReduceType for reduction operations.
const (
	ReduceSum     = 0
	ReduceProduct = 1
	ReduceMax     = 2
	ReduceMin     = 3
)

// ScatterMode for scatter operations.
const (
	ScatterModeAdd = 0
	ScatterModeMax = 1
	ScatterModeMin = 2
)

// checkRC checks an mlx-c return code and panics on error.
func checkRC(rc C.int, op string) {
	if rc != 0 {
		panic(fmt.Sprintf("mlx: %s failed with rc=%d", op, rc))
	}
}

// checkRCErr checks an mlx-c return code and returns an error.
func checkRCErr(rc C.int, op string) error {
	if rc != 0 {
		return fmt.Errorf("mlx: %s failed with rc=%d", op, rc)
	}
	return nil
}

// ===========================================================================
// Array
// ===========================================================================

// Array wraps an mlx_array handle.
type Array struct {
	handle C.mlx_array
}

func (a *Array) valid() bool {
	return C.mlx_array_ctx_is_null(a.handle) == 0
}

// NewArray creates an empty array.
func NewArray() *Array {
	a := &Array{}
	a.handle = C.mlx_array_new()
	return a
}

// NewArrayFromData creates an array from raw data.
func NewArrayFromData(data unsafe.Pointer, shape []int, dtype DType) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	if data == nil {
		// Create zeros array instead.
		rc := C.mlx_zeros(&a.handle, shapePtr, C.size_t(len(shape)), dtype, C.mlx_default_gpu_stream_new())
		checkRC(rc, "mlx_zeros")
	} else {
		a.handle = C.mlx_array_new_data(data, shapePtr, C.int(len(shape)), dtype)
	}
	runtime.KeepAlive(data)
	return a
}

// NewArrayScalarFloat32 creates a scalar float32 array.
func NewArrayScalarFloat32(val float32) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_float(C.float(val))
	return a
}

// NewArrayScalarInt32 creates a scalar int32 array.
func NewArrayScalarInt32(val int32) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_int(C.int(val))
	return a
}

// NewArrayScalarBool creates a scalar bool array.
func NewArrayScalarBool(val bool) *Array {
	a := &Array{}
	a.handle = C.mlx_array_new_bool(C.bool(val))
	return a
}

// Free releases the array.
func (a *Array) Free() {
	if a.valid() {
		C.mlx_array_free(a.handle)
		a.handle = C.mlx_array_new() // reset to empty
	}
}

// Shape returns the dimensions.
func (a *Array) Shape() []int {
	ndim := int(C.mlx_array_ndim(a.handle))
	shape := make([]int, ndim)
	cShape := C.mlx_array_shape(a.handle)
	for i := 0; i < ndim; i++ {
		shape[i] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(cShape)) + uintptr(i)*unsafe.Sizeof(C.int(0)))))
	}
	return shape
}

// NDim returns the number of dimensions.
func (a *Array) NDim() int {
	return int(C.mlx_array_ndim(a.handle))
}

// Size returns the total number of elements.
func (a *Array) Size() int {
	return int(C.mlx_array_size(a.handle))
}

// NBytes returns the total byte count.
func (a *Array) NBytes() int {
	return int(C.mlx_array_nbytes(a.handle))
}

// DType returns the element type.
func (a *Array) DType() DType {
	return C.mlx_array_dtype(a.handle)
}

// ItemSize returns the size of each element in bytes.
func (a *Array) ItemSize() int {
	return int(C.mlx_array_itemsize(a.handle))
}

// DataPtr returns a pointer to the array's data in unified memory.
// The array must be evaluated first (call Eval).
func (a *Array) DataPtr() unsafe.Pointer {
	dtype := a.DType()
	switch dtype {
	case DTypeBool:
		return unsafe.Pointer(C.mlx_array_data_bool(a.handle))
	case DTypeUint8:
		return unsafe.Pointer(C.mlx_array_data_uint8(a.handle))
	case DTypeUint16:
		return unsafe.Pointer(C.mlx_array_data_uint16(a.handle))
	case DTypeUint32:
		return unsafe.Pointer(C.mlx_array_data_uint32(a.handle))
	case DTypeUint64:
		return unsafe.Pointer(C.mlx_array_data_uint64(a.handle))
	case DTypeInt8:
		return unsafe.Pointer(C.mlx_array_data_int8(a.handle))
	case DTypeInt16:
		return unsafe.Pointer(C.mlx_array_data_int16(a.handle))
	case DTypeInt32:
		return unsafe.Pointer(C.mlx_array_data_int32(a.handle))
	case DTypeInt64:
		return unsafe.Pointer(C.mlx_array_data_int64(a.handle))
	case DTypeFloat16:
		return unsafe.Pointer(C.mlx_array_data_float16_ptr(a.handle))
	case DTypeFloat32:
		return unsafe.Pointer(C.mlx_array_data_float32(a.handle))
	case DTypeFloat64:
		return unsafe.Pointer(C.mlx_array_data_float64(a.handle))
	default:
		return unsafe.Pointer(C.mlx_array_data_float32(a.handle))
	}
}

// ===========================================================================
// Vector of Arrays (for multi-input/output ops)
// ===========================================================================

// VectorArray wraps mlx_vector_array.
type VectorArray struct {
	handle C.mlx_vector_array
}

func (v *VectorArray) valid() bool {
	return C.mlx_vector_array_ctx_is_null(v.handle) == 0
}

// NewVectorArray creates a new vector from a slice of arrays.
func NewVectorArray(arrays []*Array) *VectorArray {
	v := &VectorArray{}
	if len(arrays) == 0 {
		v.handle = C.mlx_vector_array_new()
		return v
	}
	handles := make([]C.mlx_array, len(arrays))
	for i, a := range arrays {
		handles[i] = a.handle
	}
	v.handle = C.mlx_vector_array_new_data(&handles[0], C.size_t(len(arrays)))
	return v
}

// Free releases the vector.
func (v *VectorArray) Free() {
	if v.valid() {
		C.mlx_vector_array_free(v.handle)
		v.handle = C.mlx_vector_array_new()
	}
}

// Size returns the number of arrays in the vector.
func (v *VectorArray) Size() int {
	return int(C.mlx_vector_array_size(v.handle))
}

// Get returns the array at the given index. Caller must free the returned array.
func (v *VectorArray) Get(index int) *Array {
	a := &Array{}
	rc := C.mlx_vector_array_get(&a.handle, v.handle, C.size_t(index))
	checkRC(rc, "mlx_vector_array_get")
	return a
}

// ===========================================================================
// Stream
// ===========================================================================

// Stream wraps mlx_stream.
type Stream struct {
	handle C.mlx_stream
}

func (s *Stream) valid() bool {
	return C.mlx_stream_ctx_is_null(s.handle) == 0
}

// DefaultGPUStream returns the default GPU stream.
func DefaultGPUStream() *Stream {
	s := &Stream{}
	s.handle = C.mlx_default_gpu_stream_new()
	return s
}

// DefaultCPUStream returns the default CPU stream.
func DefaultCPUStream() *Stream {
	s := &Stream{}
	s.handle = C.mlx_default_cpu_stream_new()
	return s
}

// Free releases the stream.
func (s *Stream) Free() {
	if s.valid() {
		C.mlx_stream_free(s.handle)
		s.handle = C.mlx_stream_new()
	}
}

// ===========================================================================
// Eval
// ===========================================================================

// Eval forces evaluation of arrays. MLX is lazy — this materializes results.
func Eval(arrays ...*Array) error {
	if len(arrays) == 0 {
		return nil
	}
	v := NewVectorArray(arrays)
	defer v.Free()
	rc := C.mlx_eval(v.handle)
	return checkRCErr(rc, "mlx_eval")
}

// ===========================================================================
// Device Queries
// ===========================================================================

// MetalIsAvailable returns true if Metal GPU is available.
func MetalIsAvailable() bool {
	var result C.bool
	C.mlx_metal_is_available(&result)
	return bool(result)
}

// ===========================================================================
// Memory Management
// ===========================================================================

// ClearCache frees cached GPU memory.
func ClearCache() {
	C.mlx_clear_cache()
}

// GetActiveMemory returns current GPU memory allocation in bytes.
func GetActiveMemory() uint64 {
	var res C.size_t
	C.mlx_get_active_memory(&res)
	return uint64(res)
}

// SetMemoryLimit sets the maximum GPU memory limit.
func SetMemoryLimit(limit uint64) uint64 {
	var res C.size_t
	C.mlx_set_memory_limit(&res, C.size_t(limit))
	return uint64(res)
}

// SetCacheLimit sets the maximum GPU cache limit.
func SetCacheLimit(limit uint64) uint64 {
	var res C.size_t
	C.mlx_set_cache_limit(&res, C.size_t(limit))
	return uint64(res)
}

// ===========================================================================
// Unary Operations
// ===========================================================================

func Abs(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_abs(&r.handle, x.handle, s.handle), "mlx_abs"); return r
}
func Negative(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_negative(&r.handle, x.handle, s.handle), "mlx_negative"); return r
}
func Sqrt(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sqrt(&r.handle, x.handle, s.handle), "mlx_sqrt"); return r
}
func Rsqrt(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_rsqrt(&r.handle, x.handle, s.handle), "mlx_rsqrt"); return r
}
func Exp(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_exp(&r.handle, x.handle, s.handle), "mlx_exp"); return r
}
func Expm1(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_expm1(&r.handle, x.handle, s.handle), "mlx_expm1"); return r
}
func Log(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_log(&r.handle, x.handle, s.handle), "mlx_log"); return r
}
func Log1p(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_log1p(&r.handle, x.handle, s.handle), "mlx_log1p"); return r
}
func Sin(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sin(&r.handle, x.handle, s.handle), "mlx_sin"); return r
}
func Cos(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_cos(&r.handle, x.handle, s.handle), "mlx_cos"); return r
}
func Tanh(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_tanh(&r.handle, x.handle, s.handle), "mlx_tanh"); return r
}
func Sigmoid(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sigmoid(&r.handle, x.handle, s.handle), "mlx_sigmoid"); return r
}
func Erf(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_erf(&r.handle, x.handle, s.handle), "mlx_erf"); return r
}
func Floor(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_floor(&r.handle, x.handle, s.handle), "mlx_floor"); return r
}
func Ceil(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_ceil(&r.handle, x.handle, s.handle), "mlx_ceil"); return r
}
func Sign(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_sign(&r.handle, x.handle, s.handle), "mlx_sign"); return r
}
func LogicalNot(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_not(&r.handle, x.handle, s.handle), "mlx_logical_not"); return r
}
func IsNaN(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_isnan(&r.handle, x.handle, s.handle), "mlx_isnan"); return r
}
func BitwiseNot(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_invert(&r.handle, x.handle, s.handle), "mlx_bitwise_invert"); return r
}
func Round(x *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_round(&r.handle, x.handle, 0, s.handle), "mlx_round"); return r
}

// ===========================================================================
// Binary Operations
// ===========================================================================

func Add(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_add(&r.handle, a.handle, b.handle, s.handle), "mlx_add"); return r
}
func Subtract(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_subtract(&r.handle, a.handle, b.handle, s.handle), "mlx_subtract"); return r
}
func Multiply(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_multiply(&r.handle, a.handle, b.handle, s.handle), "mlx_multiply"); return r
}
func Divide(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_divide(&r.handle, a.handle, b.handle, s.handle), "mlx_divide"); return r
}
func Remainder(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_remainder(&r.handle, a.handle, b.handle, s.handle), "mlx_remainder"); return r
}
func Power(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_power(&r.handle, a.handle, b.handle, s.handle), "mlx_power"); return r
}
func Maximum(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_maximum(&r.handle, a.handle, b.handle, s.handle), "mlx_maximum"); return r
}
func Minimum(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_minimum(&r.handle, a.handle, b.handle, s.handle), "mlx_minimum"); return r
}
func Arctan2(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_arctan2(&r.handle, a.handle, b.handle, s.handle), "mlx_arctan2"); return r
}
func LogicalAnd(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_and(&r.handle, a.handle, b.handle, s.handle), "mlx_logical_and"); return r
}
func LogicalOr(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_logical_or(&r.handle, a.handle, b.handle, s.handle), "mlx_logical_or"); return r
}
func BitwiseAnd(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_and(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_and"); return r
}
func BitwiseOr(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_or(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_or"); return r
}
func BitwiseXor(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_bitwise_xor(&r.handle, a.handle, b.handle, s.handle), "mlx_bitwise_xor"); return r
}
func LeftShift(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_left_shift(&r.handle, a.handle, b.handle, s.handle), "mlx_left_shift"); return r
}
func RightShift(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_right_shift(&r.handle, a.handle, b.handle, s.handle), "mlx_right_shift"); return r
}
func Equal(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_equal"); return r
}
func NotEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_not_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_not_equal"); return r
}
func Less(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_less(&r.handle, a.handle, b.handle, s.handle), "mlx_less"); return r
}
func LessEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_less_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_less_equal"); return r
}
func Greater(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_greater(&r.handle, a.handle, b.handle, s.handle), "mlx_greater"); return r
}
func GreaterEqual(a, b *Array, s *Stream) *Array {
	r := &Array{}; checkRC(C.mlx_greater_equal(&r.handle, a.handle, b.handle, s.handle), "mlx_greater_equal"); return r
}

// ===========================================================================
// Shape Operations
// ===========================================================================

// Reshape reshapes the array.
func Reshape(x *Array, shape []int, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_reshape(&r.handle, x.handle, shapePtr, C.size_t(len(shape)), s.handle)
	checkRC(rc, "mlx_reshape")
	return r
}

// Transpose permutes axes. Empty axes reverses all axes.
func Transpose(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	if len(axes) == 0 {
		rc := C.mlx_transpose(&r.handle, x.handle, s.handle)
		checkRC(rc, "mlx_transpose")
		return r
	}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_transpose_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_transpose_axes")
	return r
}

// BroadcastTo broadcasts to the target shape.
func BroadcastTo(x *Array, shape []int, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	rc := C.mlx_broadcast_to(&r.handle, x.handle, &cShape[0], C.size_t(len(shape)), s.handle)
	checkRC(rc, "mlx_broadcast_to")
	return r
}

// AsType casts the array to a different dtype.
func AsType(x *Array, dtype DType, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_astype(&r.handle, x.handle, dtype, s.handle)
	checkRC(rc, "mlx_astype")
	return r
}

// Concatenate joins arrays along an axis.
func Concatenate(arrays []*Array, axis int, s *Stream) *Array {
	r := &Array{}
	v := NewVectorArray(arrays)
	defer v.Free()
	rc := C.mlx_concatenate_axis(&r.handle, v.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_concatenate_axis")
	return r
}

// Slice extracts a sub-array.
func Slice(x *Array, starts, stops, strides []int, s *Stream) *Array {
	r := &Array{}
	n := len(starts)
	cStarts := make([]C.int, n)
	cStops := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStarts[i] = C.int(starts[i])
		cStops[i] = C.int(stops[i])
		cStrides[i] = C.int(strides[i])
	}
	rc := C.mlx_slice(&r.handle, x.handle, &cStarts[0], C.size_t(n), &cStops[0], C.size_t(n), &cStrides[0], C.size_t(n), s.handle)
	checkRC(rc, "mlx_slice")
	return r
}

// SliceUpdate updates a slice of an array.
func SliceUpdate(src, update *Array, starts, stops, strides []int, s *Stream) *Array {
	r := &Array{}
	n := len(starts)
	cStarts := make([]C.int, n)
	cStops := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStarts[i] = C.int(starts[i])
		cStops[i] = C.int(stops[i])
		cStrides[i] = C.int(strides[i])
	}
	rc := C.mlx_slice_update(&r.handle, src.handle, update.handle, &cStarts[0], C.size_t(n), &cStops[0], C.size_t(n), &cStrides[0], C.size_t(n), s.handle)
	checkRC(rc, "mlx_slice_update")
	return r
}

// Pad pads the array.
func Pad(x, padValue *Array, axes []int, lowPads, highPads []int, s *Stream) *Array {
	r := &Array{}
	n := len(axes)
	cAxes := make([]C.int, n)
	cLow := make([]C.int, n)
	cHigh := make([]C.int, n)
	for i := 0; i < n; i++ {
		cAxes[i] = C.int(axes[i])
		cLow[i] = C.int(lowPads[i])
		cHigh[i] = C.int(highPads[i])
	}
	cMode := C.CString("constant")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_pad(&r.handle, x.handle, &cAxes[0], C.size_t(n), &cLow[0], C.size_t(n), &cHigh[0], C.size_t(n), padValue.handle, cMode, s.handle)
	checkRC(rc, "mlx_pad")
	return r
}

// Flip reverses elements along the given axes.
func Flip(x *Array, axes []int, s *Stream) *Array {
	result := x
	shape := x.Shape()
	for _, axis := range axes {
		n := shape[axis]
		// Create reversed index array: [n-1, n-2, ..., 1, 0]
		indices := Arange(float64(n-1), -1, -1, C.MLX_INT32, s)
		result = Take(result, indices, axis, s)
	}
	return result
}

// ExpandDims adds dimensions at the given axes.
func ExpandDims(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_expand_dims_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_expand_dims_axes")
	return r
}

// Squeeze removes dimensions at the given axes.
func Squeeze(x *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_squeeze_axes(&r.handle, x.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_squeeze_axes")
	return r
}

// Where selects elements based on condition.
func Where(condition, x, y *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_where(&r.handle, condition.handle, x.handle, y.handle, s.handle)
	checkRC(rc, "mlx_where")
	return r
}

// Clip clamps values to a range.
func Clip(x, min, max *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_clip(&r.handle, x.handle, min.handle, max.handle, s.handle)
	checkRC(rc, "mlx_clip")
	return r
}

// ===========================================================================
// Reduction Operations
// ===========================================================================

func reduceSetup(axes []int) ([]C.int, *C.int, C.size_t) {
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	if len(cAxes) > 0 {
		return cAxes, &cAxes[0], C.size_t(len(axes))
	}
	return cAxes, nil, 0
}

func Sum(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_sum_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_sum_axes"); runtime.KeepAlive(ca); return r
}
func Prod(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_prod_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_prod_axes"); runtime.KeepAlive(ca); return r
}
func Max(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_max_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_max_axes"); runtime.KeepAlive(ca); return r
}
func Min(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_min_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_min_axes"); runtime.KeepAlive(ca); return r
}
func All(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_all_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_all_axes"); runtime.KeepAlive(ca); return r
}
func Any(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	r := &Array{}; ca, p, n := reduceSetup(axes)
	checkRC(C.mlx_any_axes(&r.handle, x.handle, p, n, C.bool(keepDims), s.handle), "mlx_any_axes"); runtime.KeepAlive(ca); return r
}

// ArgMin returns indices of minimum values along an axis.
func ArgMin(x *Array, axis int, keepDims bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argmin_axis(&r.handle, x.handle, C.int(axis), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_argmin_axis")
	return r
}

// ArgMax returns indices of maximum values along an axis.
func ArgMax(x *Array, axis int, keepDims bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argmax_axis(&r.handle, x.handle, C.int(axis), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_argmax_axis")
	return r
}

// ===========================================================================
// Matrix Operations
// ===========================================================================

// MatMul performs matrix multiplication.
func MatMul(lhs, rhs *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_matmul(&r.handle, lhs.handle, rhs.handle, s.handle)
	checkRC(rc, "mlx_matmul")
	return r
}

// ===========================================================================
// Gather / Scatter
// ===========================================================================

// Take gathers elements along an axis.
func Take(x, indices *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_take_axis(&r.handle, x.handle, indices.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_take_axis")
	return r
}

// TakeAlongAxis gathers elements along an axis using indices.
func TakeAlongAxis(x, indices *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_take_along_axis(&r.handle, x.handle, indices.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_take_along_axis")
	return r
}

// Gather performs generalized gather.
func Gather(x *Array, indices []*Array, axes []int, sliceSizes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	cSliceSizes := make([]C.int, len(sliceSizes))
	for i, sz := range sliceSizes {
		cSliceSizes[i] = C.int(sz)
	}
	rc := C.mlx_gather(&r.handle, x.handle, indicesVec.handle, &cAxes[0], C.size_t(len(axes)), &cSliceSizes[0], C.size_t(len(sliceSizes)), s.handle)
	checkRC(rc, "mlx_gather")
	return r
}

// ScatterAdd performs scatter with addition.
func ScatterAdd(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_add(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_add")
	return r
}

// ScatterMax performs scatter with max.
func ScatterMax(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_max(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_max")
	return r
}

// ScatterMin performs scatter with min.
func ScatterMin(x *Array, indices []*Array, updates *Array, axes []int, s *Stream) *Array {
	r := &Array{}
	indicesVec := NewVectorArray(indices)
	defer indicesVec.Free()
	cAxes := make([]C.int, len(axes))
	for i, a := range axes {
		cAxes[i] = C.int(a)
	}
	rc := C.mlx_scatter_min(&r.handle, x.handle, indicesVec.handle, updates.handle, &cAxes[0], C.size_t(len(axes)), s.handle)
	checkRC(rc, "mlx_scatter_min")
	return r
}

// ===========================================================================
// Convolution
// ===========================================================================

// Conv2d performs 2D convolution.
func Conv2d(input, weight *Array, stride, padding, dilation [2]int, groups int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_conv2d(&r.handle, input.handle, weight.handle,
		C.int(stride[0]), C.int(stride[1]),
		C.int(padding[0]), C.int(padding[1]),
		C.int(dilation[0]), C.int(dilation[1]),
		C.int(groups), s.handle)
	checkRC(rc, "mlx_conv2d")
	return r
}

// Conv1d performs 1D convolution.
func Conv1d(input, weight *Array, stride, padding, dilation, groups int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_conv1d(&r.handle, input.handle, weight.handle,
		C.int(stride), C.int(padding), C.int(dilation), C.int(groups), s.handle)
	checkRC(rc, "mlx_conv1d")
	return r
}

// ===========================================================================
// Softmax
// ===========================================================================

// Softmax computes softmax along an axis.
func Softmax(x *Array, axis int, precise bool, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_softmax_axis(&r.handle, x.handle, C.int(axis), C.bool(precise), s.handle)
	checkRC(rc, "mlx_softmax_axis")
	return r
}

// ===========================================================================
// Sort / ArgSort
// ===========================================================================

// Sort sorts along an axis.
func Sort(x *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_sort_axis(&r.handle, x.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_sort_axis")
	return r
}

// ArgSort returns sort indices along an axis.
func ArgSort(x *Array, axis int, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_argsort_axis(&r.handle, x.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_argsort_axis")
	return r
}

// ===========================================================================
// Arange (for Iota implementation)
// ===========================================================================

// Arange creates a range of values.
func Arange(start, stop, step float64, dtype DType, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_arange(&r.handle, C.double(start), C.double(stop), C.double(step), dtype, s.handle)
	checkRC(rc, "mlx_arange")
	return r
}

// ===========================================================================
// Random Number Generation
// ===========================================================================

// RandomKey creates an RNG key from a seed.
func RandomKey(seed uint64) *Array {
	r := &Array{}
	rc := C.mlx_random_key(&r.handle, C.uint64_t(seed))
	checkRC(rc, "mlx_random_key")
	return r
}

// RandomSplit splits an RNG key into two.
func RandomSplit(key *Array, s *Stream) (*Array, *Array) {
	k1 := &Array{}
	k2 := &Array{}
	rc := C.mlx_random_split(&k1.handle, &k2.handle, key.handle, s.handle)
	checkRC(rc, "mlx_random_split")
	return k1, k2
}

// RandomUniform generates uniform random values in [low, high).
func RandomUniform(low, high *Array, shape []int, dtype DType, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_uniform(&r.handle, low.handle, high.handle, shapePtr, C.size_t(len(shape)), dtype, key.handle, s.handle)
	checkRC(rc, "mlx_random_uniform")
	return r
}

// RandomNormal generates normally distributed random values.
func RandomNormal(shape []int, dtype DType, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_normal(&r.handle, shapePtr, C.size_t(len(shape)), dtype, C.float(0), C.float(1), key.handle, s.handle)
	checkRC(rc, "mlx_random_normal")
	return r
}

// RandomBits generates random bits.
func RandomBits(shape []int, width int, key *Array, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_bits(&r.handle, shapePtr, C.size_t(len(shape)), C.int(width), key.handle, s.handle)
	checkRC(rc, "mlx_random_bits")
	return r
}

// ===========================================================================
// Fast Operations
// ===========================================================================

// FastLayerNorm computes layer normalization using optimized Metal kernels.
func FastLayerNorm(x, weight, bias *Array, eps float32, s *Stream) *Array {
	r := &Array{}
	var w, b C.mlx_array
	if weight != nil {
		w = weight.handle
	}
	if bias != nil {
		b = bias.handle
	}
	rc := C.mlx_fast_layer_norm(&r.handle, x.handle, w, b, C.float(eps), s.handle)
	checkRC(rc, "mlx_fast_layer_norm")
	return r
}

// FastRMSNorm computes RMS normalization using optimized Metal kernels.
func FastRMSNorm(x, weight *Array, eps float32, s *Stream) *Array {
	r := &Array{}
	var w C.mlx_array
	if weight != nil {
		w = weight.handle
	}
	rc := C.mlx_fast_rms_norm(&r.handle, x.handle, w, C.float(eps), s.handle)
	checkRC(rc, "mlx_fast_rms_norm")
	return r
}

// FastRope computes rotary position embeddings using optimized Metal kernels.
func FastRope(x *Array, dims int, traditional bool, base float32, scale float32, offset int, s *Stream) *Array {
	r := &Array{}
	optBase := C.mlx_optional_float{value: C.float(base), has_value: true}
	var freqs C.mlx_array // null
	rc := C.mlx_fast_rope(&r.handle, x.handle, C.int(dims), C.bool(traditional),
		optBase, C.float(scale), C.int(offset), freqs, s.handle)
	checkRC(rc, "mlx_fast_rope")
	return r
}

// FastScaledDotProductAttention computes scaled dot product attention.
func FastScaledDotProductAttention(queries, keys, values, mask *Array, scale float32, s *Stream) *Array {
	r := &Array{}
	var maskArr C.mlx_array
	modeStr := "none"
	if mask != nil {
		maskArr = mask.handle
		modeStr = "additive"
	}
	maskMode := C.CString(modeStr)
	defer C.free(unsafe.Pointer(maskMode))
	var sinks C.mlx_array // null
	rc := C.mlx_fast_scaled_dot_product_attention(&r.handle,
		queries.handle, keys.handle, values.handle,
		C.float(scale), maskMode, maskArr, sinks, s.handle)
	checkRC(rc, "mlx_fast_scaled_dot_product_attention")
	return r
}

// ===========================================================================
// Quantization
// ===========================================================================

// Quantize quantizes an array.
func Quantize(w *Array, groupSize, bits int, s *Stream) (quantized, scales, biases *Array) {
	var vecRes C.mlx_vector_array
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_quantize(&vecRes, w.handle, optGroupSize, optBits, cMode, s.handle)
	checkRC(rc, "mlx_quantize")
	v := &VectorArray{handle: vecRes}
	defer v.Free()
	quantized = v.Get(0)
	scales = v.Get(1)
	biases = v.Get(2)
	return
}

// Dequantize dequantizes an array.
func Dequantize(w, scalesArr, biasesArr *Array, groupSize, bits int, s *Stream) *Array {
	r := &Array{}
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	optDtype := C.mlx_optional_dtype{has_value: false}
	rc := C.mlx_dequantize(&r.handle, w.handle, scalesArr.handle, biasesArr.handle,
		optGroupSize, optBits, cMode, optDtype, s.handle)
	checkRC(rc, "mlx_dequantize")
	return r
}

// QuantizedMatMul performs quantized matrix multiplication.
func QuantizedMatMul(x, w, scalesArr, biasesArr *Array, transpose bool, groupSize, bits int, s *Stream) *Array {
	r := &Array{}
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	cMode := C.CString("affine")
	defer C.free(unsafe.Pointer(cMode))
	rc := C.mlx_quantized_matmul(&r.handle, x.handle, w.handle, scalesArr.handle, biasesArr.handle,
		C.bool(transpose), optGroupSize, optBits, cMode, s.handle)
	checkRC(rc, "mlx_quantized_matmul")
	return r
}

// ===========================================================================
// IsInf (for IsFinite implementation)
// ===========================================================================

// IsInf checks for infinity.
func IsInf(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_isinf(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_isinf")
	return r
}

// Full creates an array filled with a scalar value.
func Full(shape []int, val *Array, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_full(&r.handle, shapePtr, C.size_t(len(shape)), val.handle, dtype, s.handle)
	checkRC(rc, "mlx_full")
	return r
}

// Zeros creates a zero-filled array.
func Zeros(shape []int, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_zeros(&r.handle, shapePtr, C.size_t(len(shape)), dtype, s.handle)
	checkRC(rc, "mlx_zeros")
	return r
}

// Ones creates a ones-filled array.
func Ones(shape []int, dtype DType, s *Stream) *Array {
	r := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_ones(&r.handle, shapePtr, C.size_t(len(shape)), dtype, s.handle)
	checkRC(rc, "mlx_ones")
	return r
}

// StopGradient prevents gradient flow through this array.
func StopGradient(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_stop_gradient(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_stop_gradient")
	return r
}

// Copy creates a copy of the array.
func Copy(x *Array, s *Stream) *Array {
	r := &Array{}
	rc := C.mlx_copy(&r.handle, x.handle, s.handle)
	checkRC(rc, "mlx_copy")
	return r
}

// AsStrided creates a view of the array with custom strides and shape.
func AsStrided(x *Array, shape []int, strides []int64, offset int, s *Stream) *Array {
	r := &Array{}
	var shapePtr *C.int
	var stridesPtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int)(unsafe.Pointer(&shape[0]))
	}
	if len(strides) > 0 {
		stridesPtr = (*C.int64_t)(unsafe.Pointer(&strides[0]))
	}
	rc := C.mlx_as_strided(&r.handle, x.handle,
		shapePtr, C.size_t(len(shape)),
		stridesPtr, C.size_t(len(strides)),
		C.size_t(offset), s.handle)
	checkRC(rc, "mlx_as_strided")
	return r
}

// ArraySet replaces the contents of dst with src, preserving the dst handle.
func ArraySet(dst, src *Array) {
	rc := C.mlx_array_set(&dst.handle, src.handle)
	checkRC(rc, "mlx_array_set")
}

// ===========================================================================
// Closure (for compiled function execution)
// ===========================================================================

// GoClosureFunc is the Go function type that closures wrap.
// It takes input arrays and returns output arrays.
type GoClosureFunc func(inputs []*Array) []*Array

// closureRegistry maps payload IDs to Go closure functions.
var closureRegistry struct {
	sync.Mutex
	funcs   map[uintptr]GoClosureFunc
	counter uintptr
}

func init() {
	closureRegistry.funcs = make(map[uintptr]GoClosureFunc)
}

//export goClosureCallback
func goClosureCallback(res *C.mlx_vector_array, inputs C.mlx_vector_array, payload unsafe.Pointer) C.int {
	id := uintptr(payload)
	closureRegistry.Lock()
	fn, ok := closureRegistry.funcs[id]
	closureRegistry.Unlock()
	if !ok {
		return 1 // error: callback not found
	}

	// Unpack input arrays from the vector.
	inVec := &VectorArray{handle: inputs}
	n := inVec.Size()
	goInputs := make([]*Array, n)
	for i := 0; i < n; i++ {
		goInputs[i] = inVec.Get(i)
	}

	// Call the Go function.
	goOutputs := fn(goInputs)

	// Pack output arrays into the result vector.
	outVec := NewVectorArray(goOutputs)
	*res = outVec.handle

	return 0
}

// Closure wraps an mlx_closure that maps inputs to outputs.
type Closure struct {
	handle     C.mlx_closure
	registryID uintptr // non-zero if registered in closureRegistry
}

// NewClosureFromGoFunc creates an MLX closure backed by a Go function.
func NewClosureFromGoFunc(fn GoClosureFunc) *Closure {
	closureRegistry.Lock()
	closureRegistry.counter++
	id := closureRegistry.counter
	closureRegistry.funcs[id] = fn
	closureRegistry.Unlock()

	cl := &Closure{
		registryID: id,
	}
	cl.handle = C.create_go_closure(unsafe.Pointer(id))
	return cl
}

// CompileClosure wraps a closure with mlx_compile for fused GPU execution.
func CompileClosure(cl *Closure, shapeless bool) *Closure {
	compiled := &Closure{}
	rc := C.mlx_compile(&compiled.handle, cl.handle, C.bool(shapeless))
	checkRC(rc, "mlx_compile")
	return compiled
}

// Free releases the closure and unregisters from the registry if needed.
func (cl *Closure) Free() {
	if cl.registryID != 0 {
		closureRegistry.Lock()
		delete(closureRegistry.funcs, cl.registryID)
		closureRegistry.Unlock()
		cl.registryID = 0
	}
	C.mlx_closure_free(cl.handle)
}

// ApplyClosure calls the closure with input arrays and returns output arrays.
func ApplyClosure(cl *Closure, inputs []*Array) ([]*Array, error) {
	inVec := NewVectorArray(inputs)
	defer inVec.Free()
	var outVec C.mlx_vector_array
	rc := C.mlx_closure_apply(&outVec, cl.handle, inVec.handle)
	if err := checkRCErr(rc, "mlx_closure_apply"); err != nil {
		return nil, err
	}
	v := &VectorArray{handle: outVec}
	defer v.Free()
	n := v.Size()
	results := make([]*Array, n)
	for i := 0; i < n; i++ {
		results[i] = v.Get(i)
	}
	return results, nil
}

// NewClosureFromCTape creates an MLX closure backed by a pure-C tape interpreter.
// This avoids CGo boundary crossings during steady-state execution.
func NewClosureFromCTape(instrs []int32, numSlots int, outputSlots, constSlots []int32, constArrays []*Array, paramSlots []int32) *Closure {
	p := (*C.tape_payload)(C.calloc(1, C.size_t(unsafe.Sizeof(C.tape_payload{}))))

	// Copy instruction stream.
	if len(instrs) > 0 {
		p.instrs = (*C.int32_t)(C.malloc(C.size_t(len(instrs)) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
		C.memcpy(unsafe.Pointer(p.instrs), unsafe.Pointer(&instrs[0]), C.size_t(len(instrs))*C.size_t(unsafe.Sizeof(C.int32_t(0))))
	}
	p.instrs_len = C.int32_t(len(instrs))
	p.num_slots = C.int32_t(numSlots)

	// Copy output slots.
	if len(outputSlots) > 0 {
		p.output_slots = (*C.int32_t)(C.malloc(C.size_t(len(outputSlots)) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
		C.memcpy(unsafe.Pointer(p.output_slots), unsafe.Pointer(&outputSlots[0]), C.size_t(len(outputSlots))*C.size_t(unsafe.Sizeof(C.int32_t(0))))
	}
	p.num_outputs = C.int32_t(len(outputSlots))

	// Copy constant arrays (retain references so C side owns them).
	if len(constSlots) > 0 {
		p.const_slots = (*C.int32_t)(C.malloc(C.size_t(len(constSlots)) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
		C.memcpy(unsafe.Pointer(p.const_slots), unsafe.Pointer(&constSlots[0]), C.size_t(len(constSlots))*C.size_t(unsafe.Sizeof(C.int32_t(0))))
		p.const_arrays = (*C.mlx_array)(C.malloc(C.size_t(len(constArrays)) * C.size_t(unsafe.Sizeof(C.mlx_array{}))))
		arrSlice := unsafe.Slice((*C.mlx_array)(unsafe.Pointer(p.const_arrays)), len(constArrays))
		for i, arr := range constArrays {
			// Create a new reference (retain) for the C side.
			var ref C.mlx_array
			C.mlx_array_set(&ref, arr.handle)
			arrSlice[i] = ref
		}
	}
	p.num_consts = C.int32_t(len(constSlots))

	// Copy param slots.
	if len(paramSlots) > 0 {
		p.param_slots = (*C.int32_t)(C.malloc(C.size_t(len(paramSlots)) * C.size_t(unsafe.Sizeof(C.int32_t(0)))))
		C.memcpy(unsafe.Pointer(p.param_slots), unsafe.Pointer(&paramSlots[0]), C.size_t(len(paramSlots))*C.size_t(unsafe.Sizeof(C.int32_t(0))))
	}
	p.num_params = C.int32_t(len(paramSlots))

	cl := &Closure{}
	cl.handle = C.create_tape_closure(p)
	return cl
}

// ===========================================================================
// Autograd Transforms: value_and_grad, vjp, jvp
// ===========================================================================

// ValueAndGradClosure wraps an mlx_closure_value_and_grad.
// It is created by ValueAndGrad and applied to inputs to get both
// the function outputs and the gradients with respect to specified arguments.
type ValueAndGradClosure struct {
	handle C.mlx_closure_value_and_grad
}

// ValueAndGrad creates a value-and-gradient closure from a function closure.
// argnums specifies which positional arguments to differentiate with respect to.
// The returned closure, when applied, computes both the function outputs and
// the gradients of the outputs with respect to the specified arguments.
func ValueAndGrad(fn *Closure, argnums []int) *ValueAndGradClosure {
	vg := &ValueAndGradClosure{}
	cArgnums := make([]C.int, len(argnums))
	for i, a := range argnums {
		cArgnums[i] = C.int(a)
	}
	var argnumsPtr *C.int
	if len(cArgnums) > 0 {
		argnumsPtr = &cArgnums[0]
	}
	rc := C.mlx_value_and_grad(&vg.handle, fn.handle, argnumsPtr, C.size_t(len(argnums)))
	checkRC(rc, "mlx_value_and_grad")
	return vg
}

// Apply evaluates the value-and-grad closure on the given inputs.
// Returns (values, grads) where values are the function outputs and
// grads are the gradients with respect to the argnums specified at creation.
func (vg *ValueAndGradClosure) Apply(inputs []*Array) (values []*Array, grads []*Array, err error) {
	inVec := NewVectorArray(inputs)
	defer inVec.Free()
	var valVec, gradVec C.mlx_vector_array
	rc := C.mlx_closure_value_and_grad_apply(&valVec, &gradVec, vg.handle, inVec.handle)
	if err := checkRCErr(rc, "mlx_closure_value_and_grad_apply"); err != nil {
		return nil, nil, err
	}
	vv := &VectorArray{handle: valVec}
	defer vv.Free()
	gv := &VectorArray{handle: gradVec}
	defer gv.Free()

	values = make([]*Array, vv.Size())
	for i := range values {
		values[i] = vv.Get(i)
	}
	grads = make([]*Array, gv.Size())
	for i := range grads {
		grads[i] = gv.Get(i)
	}
	return values, grads, nil
}

// Free releases the value-and-grad closure.
func (vg *ValueAndGradClosure) Free() {
	C.mlx_closure_value_and_grad_free(vg.handle)
}

// VJP computes the Vector-Jacobian Product (reverse-mode autodiff).
// Given a function closure, primals (input values), and cotangents (output sensitivities),
// returns (outputs, vjps) where vjps are the gradients propagated backward through the function.
func VJP(fn *Closure, primals []*Array, cotangents []*Array) (outputs []*Array, vjps []*Array, err error) {
	primVec := NewVectorArray(primals)
	defer primVec.Free()
	cotVec := NewVectorArray(cotangents)
	defer cotVec.Free()

	var outVec, vjpVec C.mlx_vector_array
	rc := C.mlx_vjp(&outVec, &vjpVec, fn.handle, primVec.handle, cotVec.handle)
	if err := checkRCErr(rc, "mlx_vjp"); err != nil {
		return nil, nil, err
	}

	ov := &VectorArray{handle: outVec}
	defer ov.Free()
	jv := &VectorArray{handle: vjpVec}
	defer jv.Free()

	outputs = make([]*Array, ov.Size())
	for i := range outputs {
		outputs[i] = ov.Get(i)
	}
	vjps = make([]*Array, jv.Size())
	for i := range vjps {
		vjps[i] = jv.Get(i)
	}
	return outputs, vjps, nil
}

// JVP computes the Jacobian-Vector Product (forward-mode autodiff).
// Given a function closure, primals (input values), and tangents (input perturbations),
// returns (outputs, jvps) where jvps are the perturbations propagated forward through the function.
func JVP(fn *Closure, primals []*Array, tangents []*Array) (outputs []*Array, jvps []*Array, err error) {
	primVec := NewVectorArray(primals)
	defer primVec.Free()
	tanVec := NewVectorArray(tangents)
	defer tanVec.Free()

	var outVec, jvpVec C.mlx_vector_array
	rc := C.mlx_jvp(&outVec, &jvpVec, fn.handle, primVec.handle, tanVec.handle)
	if err := checkRCErr(rc, "mlx_jvp"); err != nil {
		return nil, nil, err
	}

	ov := &VectorArray{handle: outVec}
	defer ov.Free()
	jv := &VectorArray{handle: jvpVec}
	defer jv.Free()

	outputs = make([]*Array, ov.Size())
	for i := range outputs {
		outputs[i] = ov.Get(i)
	}
	jvps = make([]*Array, jv.Size())
	for i := range jvps {
		jvps[i] = jv.Get(i)
	}
	return outputs, jvps, nil
}

// Checkpoint wraps a closure for gradient checkpointing (memory-efficient backprop).
// During the forward pass, intermediate values are not stored. During the backward pass,
// the function is re-evaluated to recompute intermediates, trading compute for memory.
func Checkpoint(fn *Closure) *Closure {
	cl := &Closure{}
	rc := C.mlx_checkpoint(&cl.handle, fn.handle)
	checkRC(rc, "mlx_checkpoint")
	return cl
}

// AsyncEval asynchronously evaluates the given arrays.
// Unlike Eval, this returns immediately and the computation runs in the background.
func AsyncEval(arrays ...*Array) error {
	v := NewVectorArray(arrays)
	defer v.Free()
	rc := C.mlx_async_eval(v.handle)
	return checkRCErr(rc, "mlx_async_eval")
}

// ===========================================================================
// Memory Profiling (Tier 1)
// ===========================================================================

// GetCacheMemory returns current GPU cache memory in bytes.
func GetCacheMemory() uint64 {
	var res C.size_t
	C.mlx_get_cache_memory(&res)
	return uint64(res)
}

// GetPeakMemory returns peak GPU memory usage in bytes.
func GetPeakMemory() uint64 {
	var res C.size_t
	C.mlx_get_peak_memory(&res)
	return uint64(res)
}

// ResetPeakMemory resets the peak memory counter.
func ResetPeakMemory() {
	C.mlx_reset_peak_memory()
}

// SetWiredLimit sets the wired memory limit. Returns the previous limit.
func SetWiredLimit(limit uint64) uint64 {
	var res C.size_t
	C.mlx_set_wired_limit(&res, C.size_t(limit))
	return uint64(res)
}

// ===========================================================================
// Cumulative Ops (Tier 1)
// ===========================================================================

// CumSum computes the cumulative sum along an axis.
func CumSum(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_cumsum(&a.handle, x.handle, C.int(axis), C.bool(reverse), C.bool(inclusive), s.handle)
	checkRC(rc, "mlx_cumsum")
	return a
}

// CumProd computes the cumulative product along an axis.
func CumProd(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_cumprod(&a.handle, x.handle, C.int(axis), C.bool(reverse), C.bool(inclusive), s.handle)
	checkRC(rc, "mlx_cumprod")
	return a
}

// CumMax computes the cumulative maximum along an axis.
func CumMax(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_cummax(&a.handle, x.handle, C.int(axis), C.bool(reverse), C.bool(inclusive), s.handle)
	checkRC(rc, "mlx_cummax")
	return a
}

// CumMin computes the cumulative minimum along an axis.
func CumMin(x *Array, axis int, reverse, inclusive bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_cummin(&a.handle, x.handle, C.int(axis), C.bool(reverse), C.bool(inclusive), s.handle)
	checkRC(rc, "mlx_cummin")
	return a
}

// ===========================================================================
// Statistical Ops (Tier 1)
// ===========================================================================

// Mean computes the mean over all elements.
func Mean(x *Array, keepDims bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_mean(&a.handle, x.handle, C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_mean")
	return a
}

// MeanAxes computes the mean along specified axes.
func MeanAxes(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	a := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_mean_axes(&a.handle, x.handle, axesPtr, C.size_t(len(axes)), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_mean_axes")
	return a
}

// MeanAxis computes the mean along a single axis.
func MeanAxis(x *Array, axis int, keepDims bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_mean_axis(&a.handle, x.handle, C.int(axis), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_mean_axis")
	return a
}

// Variance computes the variance over all elements.
func Variance(x *Array, keepDims bool, ddof int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_var(&a.handle, x.handle, C.bool(keepDims), C.int(ddof), s.handle)
	checkRC(rc, "mlx_var")
	return a
}

// VarianceAxes computes the variance along specified axes.
func VarianceAxes(x *Array, axes []int, keepDims bool, ddof int, s *Stream) *Array {
	a := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_var_axes(&a.handle, x.handle, axesPtr, C.size_t(len(axes)), C.bool(keepDims), C.int(ddof), s.handle)
	checkRC(rc, "mlx_var_axes")
	return a
}

// StdDev computes the standard deviation over all elements.
func StdDev(x *Array, keepDims bool, ddof int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_std(&a.handle, x.handle, C.bool(keepDims), C.int(ddof), s.handle)
	checkRC(rc, "mlx_std")
	return a
}

// StdDevAxes computes the standard deviation along specified axes.
func StdDevAxes(x *Array, axes []int, keepDims bool, ddof int, s *Stream) *Array {
	a := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_std_axes(&a.handle, x.handle, axesPtr, C.size_t(len(axes)), C.bool(keepDims), C.int(ddof), s.handle)
	checkRC(rc, "mlx_std_axes")
	return a
}

// LogSumExp computes log-sum-exp over all elements.
func LogSumExp(x *Array, keepDims bool, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_logsumexp(&a.handle, x.handle, C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_logsumexp")
	return a
}

// LogSumExpAxes computes log-sum-exp along specified axes.
func LogSumExpAxes(x *Array, axes []int, keepDims bool, s *Stream) *Array {
	a := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_logsumexp_axes(&a.handle, x.handle, axesPtr, C.size_t(len(axes)), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_logsumexp_axes")
	return a
}

// LogAddExp computes element-wise log(exp(a) + exp(b)).
func LogAddExp(a, b *Array, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_logaddexp(&out.handle, a.handle, b.handle, s.handle)
	checkRC(rc, "mlx_logaddexp")
	return out
}

// ===========================================================================
// Additional Array Ops (Tier 1)
// ===========================================================================

// TopK returns the top k elements along the last axis.
func TopK(x *Array, k int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_topk(&a.handle, x.handle, C.int(k), s.handle)
	checkRC(rc, "mlx_topk")
	return a
}

// TopKAxis returns the top k elements along a specified axis.
func TopKAxis(x *Array, k, axis int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_topk_axis(&a.handle, x.handle, C.int(k), C.int(axis), s.handle)
	checkRC(rc, "mlx_topk_axis")
	return a
}

// Flatten flattens dimensions from start_axis to end_axis.
func Flatten(x *Array, startAxis, endAxis int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_flatten(&a.handle, x.handle, C.int(startAxis), C.int(endAxis), s.handle)
	checkRC(rc, "mlx_flatten")
	return a
}

// Unflatten expands a dimension into a given shape.
func Unflatten(x *Array, axis int, shape []int, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	rc := C.mlx_unflatten(&a.handle, x.handle, C.int(axis), &cShape[0], C.size_t(len(shape)), s.handle)
	checkRC(rc, "mlx_unflatten")
	return a
}

// Diagonal extracts a diagonal from a 2D array.
func Diagonal(x *Array, offset, axis1, axis2 int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_diagonal(&a.handle, x.handle, C.int(offset), C.int(axis1), C.int(axis2), s.handle)
	checkRC(rc, "mlx_diagonal")
	return a
}

// Trace computes the trace of a matrix.
func Trace(x *Array, offset, axis1, axis2 int, dtype DType, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_trace(&a.handle, x.handle, C.int(offset), C.int(axis1), C.int(axis2), dtype, s.handle)
	checkRC(rc, "mlx_trace")
	return a
}

// Tile repeats an array along each dimension.
func Tile(x *Array, reps []int, s *Stream) *Array {
	a := &Array{}
	cReps := make([]C.int, len(reps))
	for i, r := range reps {
		cReps[i] = C.int(r)
	}
	rc := C.mlx_tile(&a.handle, x.handle, &cReps[0], C.size_t(len(reps)), s.handle)
	checkRC(rc, "mlx_tile")
	return a
}

// Repeat repeats elements of an array (flattened).
func Repeat(x *Array, repeats int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_repeat(&a.handle, x.handle, C.int(repeats), s.handle)
	checkRC(rc, "mlx_repeat")
	return a
}

// RepeatAxis repeats elements of an array along an axis.
func RepeatAxis(x *Array, repeats, axis int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_repeat_axis(&a.handle, x.handle, C.int(repeats), C.int(axis), s.handle)
	checkRC(rc, "mlx_repeat_axis")
	return a
}

// Split splits an array into equal sections along an axis.
func Split(x *Array, numSplits, axis int, s *Stream) *VectorArray {
	v := &VectorArray{}
	rc := C.mlx_split(&v.handle, x.handle, C.int(numSplits), C.int(axis), s.handle)
	checkRC(rc, "mlx_split")
	return v
}

// SplitSections splits an array at given indices along an axis.
func SplitSections(x *Array, indices []int, axis int, s *Stream) *VectorArray {
	v := &VectorArray{}
	cIndices := make([]C.int, len(indices))
	for i, idx := range indices {
		cIndices[i] = C.int(idx)
	}
	rc := C.mlx_split_sections(&v.handle, x.handle, &cIndices[0], C.size_t(len(indices)), C.int(axis), s.handle)
	checkRC(rc, "mlx_split_sections")
	return v
}

// Linspace generates evenly spaced numbers.
func Linspace(start, stop float64, num int, dtype DType, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_linspace(&a.handle, C.double(start), C.double(stop), C.int(num), dtype, s.handle)
	checkRC(rc, "mlx_linspace")
	return a
}

// AddMM computes alpha * (a @ b) + beta * c.
func AddMM(c, a, b *Array, alpha, beta float32, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_addmm(&out.handle, c.handle, a.handle, b.handle, C.float(alpha), C.float(beta), s.handle)
	checkRC(rc, "mlx_addmm")
	return out
}

// ===========================================================================
// Advanced RNG (Tier 2)
// ===========================================================================

// RandomBernoulli generates Bernoulli random samples.
func RandomBernoulli(p *Array, shape []int, key *Array, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_bernoulli(&a.handle, p.handle, shapePtr, C.size_t(len(shape)), key.handle, s.handle)
	checkRC(rc, "mlx_random_bernoulli")
	return a
}

// RandomCategorical samples from a categorical distribution.
func RandomCategorical(logits *Array, axis int, key *Array, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_random_categorical(&a.handle, logits.handle, C.int(axis), key.handle, s.handle)
	checkRC(rc, "mlx_random_categorical")
	return a
}

// RandomCategoricalNumSamples samples from a categorical distribution with num_samples.
func RandomCategoricalNumSamples(logits *Array, axis, numSamples int, key *Array, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_random_categorical_num_samples(&a.handle, logits.handle, C.int(axis), C.int(numSamples), key.handle, s.handle)
	checkRC(rc, "mlx_random_categorical_num_samples")
	return a
}

// RandomPermutation returns a random permutation of an array along an axis.
func RandomPermutation(x *Array, axis int, key *Array, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_random_permutation(&a.handle, x.handle, C.int(axis), key.handle, s.handle)
	checkRC(rc, "mlx_random_permutation")
	return a
}

// RandomPermutationArange returns a random permutation of [0, n).
func RandomPermutationArange(n int, key *Array, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_random_permutation_arange(&a.handle, C.int(n), key.handle, s.handle)
	checkRC(rc, "mlx_random_permutation_arange")
	return a
}

// RandomRandInt generates uniform random integers in [low, high).
func RandomRandInt(low, high *Array, shape []int, dtype DType, key *Array, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_randint(&a.handle, low.handle, high.handle, shapePtr, C.size_t(len(shape)), dtype, key.handle, s.handle)
	checkRC(rc, "mlx_random_randint")
	return a
}

// RandomGumbel generates Gumbel random samples.
func RandomGumbel(shape []int, dtype DType, key *Array, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_gumbel(&a.handle, shapePtr, C.size_t(len(shape)), dtype, key.handle, s.handle)
	checkRC(rc, "mlx_random_gumbel")
	return a
}

// RandomLaplace generates Laplace random samples.
func RandomLaplace(shape []int, dtype DType, loc, scale float32, key *Array, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_laplace(&a.handle, shapePtr, C.size_t(len(shape)), dtype, C.float(loc), C.float(scale), key.handle, s.handle)
	checkRC(rc, "mlx_random_laplace")
	return a
}

// RandomTruncatedNormal generates truncated normal random samples.
func RandomTruncatedNormal(lower, upper *Array, shape []int, dtype DType, key *Array, s *Stream) *Array {
	a := &Array{}
	cShape := make([]C.int, len(shape))
	for i, d := range shape {
		cShape[i] = C.int(d)
	}
	var shapePtr *C.int
	if len(cShape) > 0 {
		shapePtr = &cShape[0]
	}
	rc := C.mlx_random_truncated_normal(&a.handle, lower.handle, upper.handle, shapePtr, C.size_t(len(shape)), dtype, key.handle, s.handle)
	checkRC(rc, "mlx_random_truncated_normal")
	return a
}

// RandomSeed sets the global random seed.
func RandomSeed(seed uint64) {
	C.mlx_random_seed(C.uint64_t(seed))
}

// ===========================================================================
// Linear Algebra (Tier 2)
// ===========================================================================

// LinalgCholesky computes the Cholesky decomposition.
func LinalgCholesky(a *Array, upper bool, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_cholesky(&out.handle, a.handle, C.bool(upper), s.handle)
	checkRC(rc, "mlx_linalg_cholesky")
	return out
}

// LinalgInv computes the matrix inverse.
func LinalgInv(a *Array, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_inv(&out.handle, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_inv")
	return out
}

// LinalgSolve solves a linear system Ax = b.
func LinalgSolve(a, b *Array, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_solve(&out.handle, a.handle, b.handle, s.handle)
	checkRC(rc, "mlx_linalg_solve")
	return out
}

// LinalgSolveTriangular solves a triangular linear system.
func LinalgSolveTriangular(a, b *Array, upper bool, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_solve_triangular(&out.handle, a.handle, b.handle, C.bool(upper), s.handle)
	checkRC(rc, "mlx_linalg_solve_triangular")
	return out
}

// LinalgSVD computes the singular value decomposition.
// Returns [U, S, Vt] as a vector array.
func LinalgSVD(a *Array, computeUV bool, s *Stream) *VectorArray {
	v := &VectorArray{}
	rc := C.mlx_linalg_svd(&v.handle, a.handle, C.bool(computeUV), s.handle)
	checkRC(rc, "mlx_linalg_svd")
	return v
}

// LinalgQR computes the QR decomposition. Returns (Q, R).
func LinalgQR(a *Array, s *Stream) (*Array, *Array) {
	var q, r C.mlx_array
	rc := C.mlx_linalg_qr(&q, &r, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_qr")
	return &Array{handle: q}, &Array{handle: r}
}

// LinalgEig computes eigenvalues and eigenvectors. Returns (eigenvalues, eigenvectors).
func LinalgEig(a *Array, s *Stream) (*Array, *Array) {
	var vals, vecs C.mlx_array
	rc := C.mlx_linalg_eig(&vals, &vecs, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_eig")
	return &Array{handle: vals}, &Array{handle: vecs}
}

// LinalgEigh computes eigenvalues/vectors of a Hermitian matrix. Returns (eigenvalues, eigenvectors).
func LinalgEigh(a *Array, uplo string, s *Stream) (*Array, *Array) {
	cUplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cUplo))
	var vals, vecs C.mlx_array
	rc := C.mlx_linalg_eigh(&vals, &vecs, a.handle, cUplo, s.handle)
	checkRC(rc, "mlx_linalg_eigh")
	return &Array{handle: vals}, &Array{handle: vecs}
}

// LinalgEigvals computes eigenvalues only.
func LinalgEigvals(a *Array, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_eigvals(&out.handle, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_eigvals")
	return out
}

// LinalgNorm computes the matrix/vector norm.
func LinalgNorm(a *Array, ord float64, axes []int, keepDims bool, s *Stream) *Array {
	out := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_linalg_norm(&out.handle, a.handle, C.double(ord), axesPtr, C.size_t(len(axes)), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_linalg_norm")
	return out
}

// LinalgNormL2 computes the L2 norm.
func LinalgNormL2(a *Array, axes []int, keepDims bool, s *Stream) *Array {
	out := &Array{}
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	var axesPtr *C.int
	if len(cAxes) > 0 {
		axesPtr = &cAxes[0]
	}
	rc := C.mlx_linalg_norm_l2(&out.handle, a.handle, axesPtr, C.size_t(len(axes)), C.bool(keepDims), s.handle)
	checkRC(rc, "mlx_linalg_norm_l2")
	return out
}

// LinalgPinv computes the pseudo-inverse.
func LinalgPinv(a *Array, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_pinv(&out.handle, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_pinv")
	return out
}

// LinalgLU computes the LU decomposition. Returns a vector of [P, L, U].
func LinalgLU(a *Array, s *Stream) *VectorArray {
	v := &VectorArray{}
	rc := C.mlx_linalg_lu(&v.handle, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_lu")
	return v
}

// LinalgLUFactor computes LU factorization. Returns (LU, pivots).
func LinalgLUFactor(a *Array, s *Stream) (*Array, *Array) {
	var lu, pivots C.mlx_array
	rc := C.mlx_linalg_lu_factor(&lu, &pivots, a.handle, s.handle)
	checkRC(rc, "mlx_linalg_lu_factor")
	return &Array{handle: lu}, &Array{handle: pivots}
}

// LinalgCross computes the cross product of two arrays along an axis.
func LinalgCross(a, b *Array, axis int, s *Stream) *Array {
	out := &Array{}
	rc := C.mlx_linalg_cross(&out.handle, a.handle, b.handle, C.int(axis), s.handle)
	checkRC(rc, "mlx_linalg_cross")
	return out
}

// ===========================================================================
// Transpose Convolutions (Tier 2)
// ===========================================================================

// ConvTranspose1d computes a 1D transposed convolution.
func ConvTranspose1d(input, weight *Array, stride, padding, dilation, outputPadding, groups int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_conv_transpose1d(&a.handle, input.handle, weight.handle,
		C.int(stride), C.int(padding), C.int(dilation), C.int(outputPadding), C.int(groups), s.handle)
	checkRC(rc, "mlx_conv_transpose1d")
	return a
}

// ConvTranspose2d computes a 2D transposed convolution.
func ConvTranspose2d(input, weight *Array, stride, padding, dilation, outputPadding [2]int, groups int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_conv_transpose2d(&a.handle, input.handle, weight.handle,
		C.int(stride[0]), C.int(stride[1]),
		C.int(padding[0]), C.int(padding[1]),
		C.int(dilation[0]), C.int(dilation[1]),
		C.int(outputPadding[0]), C.int(outputPadding[1]),
		C.int(groups), s.handle)
	checkRC(rc, "mlx_conv_transpose2d")
	return a
}

// ConvTranspose3d computes a 3D transposed convolution.
func ConvTranspose3d(input, weight *Array, stride, padding, dilation, outputPadding [3]int, groups int, s *Stream) *Array {
	a := &Array{}
	rc := C.mlx_conv_transpose3d(&a.handle, input.handle, weight.handle,
		C.int(stride[0]), C.int(stride[1]), C.int(stride[2]),
		C.int(padding[0]), C.int(padding[1]), C.int(padding[2]),
		C.int(dilation[0]), C.int(dilation[1]), C.int(dilation[2]),
		C.int(outputPadding[0]), C.int(outputPadding[1]), C.int(outputPadding[2]),
		C.int(groups), s.handle)
	checkRC(rc, "mlx_conv_transpose3d")
	return a
}

// ===========================================================================
// SafeTensors IO (Tier 1)
// ===========================================================================

// MapStringToArray wraps mlx_map_string_to_array for SafeTensors I/O.
type MapStringToArray struct {
	handle C.mlx_map_string_to_array
}

// NewMapStringToArray creates a new empty string-to-array map.
func NewMapStringToArray() *MapStringToArray {
	return &MapStringToArray{handle: C.mlx_map_string_to_array_new()}
}

// Free releases the map.
func (m *MapStringToArray) Free() {
	C.mlx_map_string_to_array_free(m.handle)
}

// Insert adds a key-value pair.
func (m *MapStringToArray) Insert(key string, value *Array) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	C.mlx_map_string_to_array_insert(m.handle, cKey, value.handle)
}

// Get retrieves an array by key. Returns nil if not found.
func (m *MapStringToArray) Get(key string) *Array {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	a := &Array{}
	rc := C.mlx_map_string_to_array_get(&a.handle, m.handle, cKey)
	if rc != 0 {
		return nil
	}
	return a
}

// Iterate calls fn for each key-value pair in the map.
func (m *MapStringToArray) Iterate(fn func(key string, value *Array)) {
	it := C.mlx_map_string_to_array_iterator_new(m.handle)
	defer C.mlx_map_string_to_array_iterator_free(it)
	for {
		var cKey *C.char
		var arr C.mlx_array
		rc := C.mlx_map_string_to_array_iterator_next(&cKey, &arr, it)
		if rc != 0 {
			break
		}
		fn(C.GoString(cKey), &Array{handle: arr})
	}
}

// MapStringToString wraps mlx_map_string_to_string for SafeTensors metadata.
type MapStringToString struct {
	handle C.mlx_map_string_to_string
}

// NewMapStringToString creates a new empty string-to-string map.
func NewMapStringToString() *MapStringToString {
	return &MapStringToString{handle: C.mlx_map_string_to_string_new()}
}

// Free releases the map.
func (m *MapStringToString) Free() {
	C.mlx_map_string_to_string_free(m.handle)
}

// Insert adds a key-value pair.
func (m *MapStringToString) Insert(key, value string) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(value)
	defer C.free(unsafe.Pointer(cVal))
	C.mlx_map_string_to_string_insert(m.handle, cKey, cVal)
}

// Get retrieves a string by key. Returns ("", false) if not found.
func (m *MapStringToString) Get(key string) (string, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var cVal *C.char
	rc := C.mlx_map_string_to_string_get(&cVal, m.handle, cKey)
	if rc != 0 {
		return "", false
	}
	return C.GoString(cVal), true
}

// Iterate calls fn for each key-value pair in the map.
func (m *MapStringToString) Iterate(fn func(key, value string)) {
	it := C.mlx_map_string_to_string_iterator_new(m.handle)
	defer C.mlx_map_string_to_string_iterator_free(it)
	for {
		var cKey, cVal *C.char
		rc := C.mlx_map_string_to_string_iterator_next(&cKey, &cVal, it)
		if rc != 0 {
			break
		}
		fn(C.GoString(cKey), C.GoString(cVal))
	}
}

// LoadSafeTensors loads arrays and metadata from a SafeTensors file.
// Returns (arrays map, metadata map, error).
func LoadSafeTensors(path string, s *Stream) (*MapStringToArray, *MapStringToString, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	arrays := &MapStringToArray{}
	metadata := &MapStringToString{}
	rc := C.mlx_load_safetensors(&arrays.handle, &metadata.handle, cPath, s.handle)
	if err := checkRCErr(rc, "mlx_load_safetensors"); err != nil {
		return nil, nil, err
	}
	return arrays, metadata, nil
}

// SaveSafeTensors saves arrays and metadata to a SafeTensors file.
func SaveSafeTensors(path string, arrays *MapStringToArray, metadata *MapStringToString) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	rc := C.mlx_save_safetensors(cPath, arrays.handle, metadata.handle)
	return checkRCErr(rc, "mlx_save_safetensors")
}

// LoadArray loads a single array from a file (npy/npz format).
func LoadArray(path string, s *Stream) (*Array, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	a := &Array{}
	rc := C.mlx_load(&a.handle, cPath, s.handle)
	if err := checkRCErr(rc, "mlx_load"); err != nil {
		return nil, err
	}
	return a, nil
}

// SaveArray saves a single array to a file (npy format).
func SaveArray(path string, a *Array) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	rc := C.mlx_save(cPath, a.handle)
	return checkRCErr(rc, "mlx_save")
}
