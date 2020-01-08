//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

// VS compiler treats __VA_ARGS__ as a single token argument rather than expanding it
// so the macro below is a workaround to that
#define EXPAND_MACRO(S) S // VS compiler workaround

// Selector Macros for builders to instantiate and pick kernels
// All element types, ranks. Use for small/simple kernels
#define SELECT_KERNEL(KV, ET, K) EXPAND_ET11_FIXED_ARGS(K, KV, ET, KERNEL_CT)
#define SELECT_KERNEL_3ARGS(KV, ET, K) EXPAND_ET11_FIXED_ARGS(K, KV, ET, KERNEL_CT_CT_CT)
#define SELECT_KERNEL_RANK(KV, CIT, COT, R, K) EXPAND_RANK7(K, KV, R, KERNEL_CIT_COT_R, CIT, COT)
#define SELECT_KERNEL_ET_RANK(KV, ET, R, K) EXPAND_ET11_AND_RANK7(K, KV, ET, R, KERNEL_CT_R)

// Subset of element types and ranks. Use for more complex/larger kernels
#define SELECT_RANK35_ET4(KV, ET, R1, R2, K)                                                       \
    EXPAND_RANK35_AND_ET4(K, KV, R1, R2, ET, KERNEL_CT_R1_R2)

// Configurable at build using NGRAPH_CPU_OPTIMIZED_DATATYPES
#define SELECT_ETS(KV, ET, K) EXPAND_ETS(K, KV, ET, KERNEL_CT)
#define SELECT_ETS_AND_RANK7(KV, ET, R, K) EXPAND_ETS_AND_RANK7(K, KV, ET, R, KERNEL_CT_R)

// Macros for instantiating templated kernels
#define KERNEL_CT(K, KV, CT) KV = K<CT>
#define KERNEL_CT_CT_CT(K, KV, CT) KV = K<CT, CT, CT>
#define KERNEL_CT_R(K, KV, CT, R) KV = K<CT, R>
#define KERNEL_CIT_COT_R(K, KV, CIT, COT, R) KV = K<CIT, COT, R>
#define KERNEL_CT_R1_R2(K, KV, R1, R2, CT) KV = K<CT, R1, R2>

// Helper macros
#define EXPAND_ET11_AND_RANK7(K, KV, ET, R, S) EXPAND_ET11(K, KV, ET, EXPAND_RANK7_1, R, S)
#define EXPAND_RANK5_AND_ET4(K, KV, R, ET, S, A1) EXPAND_RANK5(K, KV, R, EXPAND_ET4, ET, S, A1)
#define EXPAND_RANK35_AND_ET4(K, KV, R1, R2, ET, S)                                                \
    EXPAND_RANK3(K, KV, R1, EXPAND_RANK5_AND_ET4, R2, ET, S)
#define EXPAND_ETS_AND_RANK7(K, KV, ET, R, S) EXPAND_ETS_2(K, KV, ET, EXPAND_RANK7_1, R, S)

// Expander Macros that instantiate kernels for various element types and ranks
#define EXPAND_ET4(K, KV, ET, S, A1, A2)                                                           \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, float));                                                     \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, double));                                                    \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, int8_t));                                                    \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, uint8_t));                                                   \
    }                                                                                              \
    else                                                                                           \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);

#define EXPAND_ET11(K, KV, ET, S, A1, A2)                                                          \
    if (ET == element::boolean)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, char));                                                      \
    }                                                                                              \
    else if (ET == element::f32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, float));                                                     \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, double));                                                    \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, int8_t));                                                    \
    }                                                                                              \
    else if (ET == element::i16)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, int16_t));                                                   \
    }                                                                                              \
    else if (ET == element::i32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, int32_t));                                                   \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, int64_t));                                                   \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, uint8_t));                                                   \
    }                                                                                              \
    else if (ET == element::u16)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, uint16_t));                                                  \
    }                                                                                              \
    else if (ET == element::u32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, uint32_t));                                                  \
    }                                                                                              \
    else if (ET == element::u64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, A1, A2, uint64_t));                                                  \
    }                                                                                              \
    else                                                                                           \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);

// Workaround since VS compiler doesn;t work well with variadic macros.
// EXPAND_ET11 Takes variable arguments and since SELECT_KERNEL & SELECT_KERNEL_3ARGS
// call into that macro without variable args, the VS compiler expands them as
// KERNEL_CT(K, KV, ,) thus giving a syntax error. The VS compiler doesn't deal well with
// igonring comma. Hence we have replicated EXPAND_ET11 and added EXPAND_ET11_FIXED_ARGS
// for calls that dont have variable args.
#define EXPAND_ET11_FIXED_ARGS(K, KV, ET, S)                                                       \
    if (ET == element::boolean)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, char));                                                              \
    }                                                                                              \
    else if (ET == element::f32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, float));                                                             \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, double));                                                            \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, int8_t));                                                            \
    }                                                                                              \
    else if (ET == element::i16)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, int16_t));                                                           \
    }                                                                                              \
    else if (ET == element::i32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, int32_t));                                                           \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, int64_t));                                                           \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, uint8_t));                                                           \
    }                                                                                              \
    else if (ET == element::u16)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, uint16_t));                                                          \
    }                                                                                              \
    else if (ET == element::u32)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, uint32_t));                                                          \
    }                                                                                              \
    else if (ET == element::u64)                                                                   \
    {                                                                                              \
        EXPAND_MACRO(S(K, KV, uint64_t));                                                          \
    }                                                                                              \
    else                                                                                           \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);

// Expand only selected datatypes. Named macros (e.g., F32_SELECT) are expanded based on build-flags
#define EXPAND_ETS(K, KV, ET, S)                                                                   \
    if (BOOLEAN_EN && ET == element::boolean)                                                      \
    {                                                                                              \
        BOOLEAN_SELECT(S, K, KV, char);                                                            \
    }                                                                                              \
    else if (F32_EN && ET == element::f32)                                                         \
    {                                                                                              \
        F32_SELECT(S, K, KV, float);                                                               \
    }                                                                                              \
    else if (F64_EN && ET == element::f64)                                                         \
    {                                                                                              \
        F64_SELECT(S, K, KV, double);                                                              \
    }                                                                                              \
    else if (I8_EN && ET == element::i8)                                                           \
    {                                                                                              \
        I8_SELECT(S, K, KV, int8_t);                                                               \
    }                                                                                              \
    else if (I16_EN && ET == element::i16)                                                         \
    {                                                                                              \
        I16_SELECT(S, K, KV, int16_t);                                                             \
    }                                                                                              \
    else if (I32_EN && ET == element::i32)                                                         \
    {                                                                                              \
        I32_SELECT(S, K, KV, int32_t);                                                             \
    }                                                                                              \
    else if (I64_EN && ET == element::i64)                                                         \
    {                                                                                              \
        I64_SELECT(S, K, KV, int64_t);                                                             \
    }                                                                                              \
    else if (U8_EN && ET == element::u8)                                                           \
    {                                                                                              \
        U8_SELECT(S, K, KV, uint8_t);                                                              \
    }                                                                                              \
    else if (U16_EN && ET == element::u16)                                                         \
    {                                                                                              \
        U16_SELECT(S, K, KV, uint16_t);                                                            \
    }                                                                                              \
    else if (U32_EN && ET == element::u32)                                                         \
    {                                                                                              \
        U32_SELECT(S, K, KV, uint32_t);                                                            \
    }                                                                                              \
    else if (U64_EN && ET == element::u64)                                                         \
    {                                                                                              \
        U64_SELECT(S, K, KV, uint64_t);                                                            \
    }                                                                                              \
    else                                                                                           \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);

#define EXPAND_ETS_2(K, KV, ET, S, A1, A2)                                                         \
    if (BOOLEAN_EN && ET == element::boolean)                                                      \
    {                                                                                              \
        BOOLEAN_SELECT(S, K, KV, A1, A2, char);                                                    \
    }                                                                                              \
    else if (F32_EN && ET == element::f32)                                                         \
    {                                                                                              \
        F32_SELECT(S, K, KV, A1, A2, float);                                                       \
    }                                                                                              \
    else if (F64_EN && ET == element::f64)                                                         \
    {                                                                                              \
        F64_SELECT(S, K, KV, A1, A2, double);                                                      \
    }                                                                                              \
    else if (I8_EN && ET == element::i8)                                                           \
    {                                                                                              \
        I8_SELECT(S, K, KV, A1, A2, int8_t);                                                       \
    }                                                                                              \
    else if (I16_EN && ET == element::i16)                                                         \
    {                                                                                              \
        I16_SELECT(S, K, KV, A1, A2, int16_t);                                                     \
    }                                                                                              \
    else if (I32_EN && ET == element::i32)                                                         \
    {                                                                                              \
        I32_SELECT(S, K, KV, A1, A2, int32_t);                                                     \
    }                                                                                              \
    else if (I64_EN && ET == element::i64)                                                         \
    {                                                                                              \
        I64_SELECT(S, K, KV, A1, A2, int64_t);                                                     \
    }                                                                                              \
    else if (U8_EN && ET == element::u8)                                                           \
    {                                                                                              \
        U8_SELECT(S, K, KV, A1, A2, uint8_t);                                                      \
    }                                                                                              \
    else if (U16_EN && ET == element::u16)                                                         \
    {                                                                                              \
        U16_SELECT(S, K, KV, A1, A2, uint16_t);                                                    \
    }                                                                                              \
    else if (U32_EN && ET == element::u32)                                                         \
    {                                                                                              \
        U32_SELECT(S, K, KV, A1, A2, uint32_t);                                                    \
    }                                                                                              \
    else if (U64_EN && ET == element::u64)                                                         \
    {                                                                                              \
        U64_SELECT(S, K, KV, A1, A2, uint64_t);                                                    \
    }                                                                                              \
    else                                                                                           \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);

#define EXPAND_RANK3(K, KV, R, S, A1, A2, A3)                                                      \
    switch (R)                                                                                     \
    {                                                                                              \
    case 1: EXPAND_MACRO(S(K, KV, A1, A2, A3, 1)); break;                                          \
    case 2: EXPAND_MACRO(S(K, KV, A1, A2, A3, 2)); break;                                          \
    case 3: EXPAND_MACRO(S(K, KV, A1, A2, A3, 3)); break;                                          \
    default: throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);      \
    }

#define EXPAND_RANK5(K, KV, R, S, A1, A2, A3)                                                      \
    switch (R)                                                                                     \
    {                                                                                              \
    case 1: EXPAND_MACRO(S(K, KV, A1, A2, A3, 1)); break;                                          \
    case 2: EXPAND_MACRO(S(K, KV, A1, A2, A3, 2)); break;                                          \
    case 3: EXPAND_MACRO(S(K, KV, A1, A2, A3, 3)); break;                                          \
    case 4: EXPAND_MACRO(S(K, KV, A1, A2, A3, 4)); break;                                          \
    case 5: EXPAND_MACRO(S(K, KV, A1, A2, A3, 5)); break;                                          \
    default: throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);      \
    }

#define EXPAND_RANK7(K, KV, R, S, A1, A2)                                                          \
    switch (R)                                                                                     \
    {                                                                                              \
    case 1: EXPAND_MACRO(S(K, KV, A1, A2, 1)); break;                                              \
    case 2: EXPAND_MACRO(S(K, KV, A1, A2, 2)); break;                                              \
    case 3: EXPAND_MACRO(S(K, KV, A1, A2, 3)); break;                                              \
    case 4: EXPAND_MACRO(S(K, KV, A1, A2, 4)); break;                                              \
    case 5: EXPAND_MACRO(S(K, KV, A1, A2, 5)); break;                                              \
    case 6: EXPAND_MACRO(S(K, KV, A1, A2, 6)); break;                                              \
    case 7: EXPAND_MACRO(S(K, KV, A1, A2, 7)); break;                                              \
    default: throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);      \
    }
#define EXPAND_RANK7_1(K, KV, R, S, A1)                                                            \
    switch (R)                                                                                     \
    {                                                                                              \
    case 1: EXPAND_MACRO(S(K, KV, A1, 1)); break;                                                  \
    case 2: EXPAND_MACRO(S(K, KV, A1, 2)); break;                                                  \
    case 3: EXPAND_MACRO(S(K, KV, A1, 3)); break;                                                  \
    case 4: EXPAND_MACRO(S(K, KV, A1, 4)); break;                                                  \
    case 5: EXPAND_MACRO(S(K, KV, A1, 5)); break;                                                  \
    case 6: EXPAND_MACRO(S(K, KV, A1, 6)); break;                                                  \
    case 7: EXPAND_MACRO(S(K, KV, A1, 7)); break;                                                  \
    default: throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);      \
    }

#if defined(NGRAPH_CPU_OPTIMIZE_boolean)
#define BOOLEAN_EN 1
#define BOOLEAN_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define BOOLEAN_EN 0
#define BOOLEAN_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_f32)
#define F32_EN 1
#define F32_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define F32_EN 0
#define F32_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_f64)
#define F64_EN 1
#define F64_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define F64_EN 0
#define F64_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_i8)
#define I8_EN 1
#define I8_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define I8_EN 0
#define I8_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_i16)
#define I16_EN 1
#define I16_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define I16_EN 0
#define I16_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_i32)
#define I32_EN 1
#define I32_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define I32_EN 0
#define I32_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_i64)
#define I64_EN 1
#define I64_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define I64_EN 0
#define I64_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_u8)
#define U8_EN 1
#define U8_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define U8_EN 0
#define U8_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_u16)
#define U16_EN 1
#define U16_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define U16_EN 0
#define U16_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_u32)
#define U32_EN 1
#define U32_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define U32_EN 0
#define U32_SELECT(S, ...)
#endif

#if defined(NGRAPH_CPU_OPTIMIZE_u64)
#define U64_EN 1
#define U64_SELECT(S, ...) EXPAND_MACRO(S(__VA_ARGS__))
#else
#define U64_EN 0
#define U64_SELECT(S, ...)
#endif

static inline bool is_optimized_et(const ngraph::element::Type& et)
{
    if ((et == ngraph::element::boolean && BOOLEAN_EN) || (et == ngraph::element::f32 && F32_EN) ||
        (et == ngraph::element::f64 && F64_EN) || (et == ngraph::element::i8 && I8_EN) ||
        (et == ngraph::element::i16 && I16_EN) || (et == ngraph::element::i32 && I32_EN) ||
        (et == ngraph::element::i64 && I64_EN) || (et == ngraph::element::u8 && U8_EN) ||
        (et == ngraph::element::u16 && U16_EN) || (et == ngraph::element::u32 && U32_EN) ||
        (et == ngraph::element::u64 && U64_EN))
    {
        return true;
    }
    else
    {
        return false;
    }
}
