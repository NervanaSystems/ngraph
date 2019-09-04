//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

#if defined(NGRAPH_CPU_ALL_DATA_TYPE_ENABLE)
#define SELECT_KERNEL_FOR_LIMITED_ET SELECT_KERNEL
#define SELECT_BY_RANK SELECT_KERNEL_BY_RANK
#define PARTIAL_SELECT_BY_RANK PARTIAL_SELECT_KERNEL_BY_RANK
#define SELECT_BY_2RANKS SELECT_KERNEL_BY_2RANKS
#endif

static inline bool is_float_or_integer_64(const ngraph::element::Type& et)
{
#if defined(NGRAPH_CPU_ALL_DATA_TYPE_ENABLE)
    return true;
#endif

    if (et == ngraph::element::f32 || et == ngraph::element::i64)
    {
        return true;
    }
    return false;
}

#define SELECT_KERNEL_FOR_LIMITED_ET(KV, ET, K)                                                    \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        KV = K<float>;                                                                             \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t>;                                                                           \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define PARTIAL_SELECT_BY_RANK(KV, ET, R, K)                                                       \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, float, R, K);                                                      \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, int64_t, R, K);                                                    \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define SELECT_BY_RANK(KV, ET, R, K)                                                               \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        SELECT_RANK(KV, float, R, K);                                                              \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, int64_t, R, K);                                                            \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define SELECT_BY_2RANKS(KV, ET, R1, R2, K)                                                        \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        SELECT_2RANKS(KV, float, R1, R2, K);                                                       \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        SELECT_2RANKS(KV, int64_t, R1, R2, K);                                                     \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }
