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

#define BUILD_REDUCTION_FUNCTOR(OP, K)                                                             \
    auto& functors = external_function->get_functors();                                            \
                                                                                                   \
    auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());               \
    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());                \
                                                                                                   \
    auto op = static_cast<const ngraph::op::OP*>(node);                                            \
                                                                                                   \
    auto arg_shape = args[0].get_shape();                                                          \
    auto arg_rank = arg_shape.size();                                                              \
                                                                                                   \
    auto result_shape = out[0].get_shape();                                                        \
    auto& result_element_type = out[0].get_element_type();                                         \
                                                                                                   \
    auto reduction_axes = op->get_reduction_axes();                                                \
                                                                                                   \
    if (reduction_axes.empty())                                                                    \
    {                                                                                              \
        size_t size = out[0].get_size() * out[0].get_element_type().size();                        \
        auto functor = [&, size, arg_buffer_index, out_buffer_index](CPURuntimeContext* ctx,       \
                                                                     CPUExecutionContext*) {       \
            memcpy(ctx->buffer_data[out_buffer_index], ctx->buffer_data[arg_buffer_index], size);  \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    if (reduction_axes.size() == arg_rank && is_optimized_et(args[0].get_element_type()))          \
    {                                                                                              \
        std::function<decltype(runtime::cpu::kernel::reduce_##K##_all<float, 2>)> kernel;          \
        SELECT_ETS_AND_RANK7(                                                                      \
            kernel, result_element_type, arg_rank, runtime::cpu::kernel::reduce_##K##_all);        \
        auto functor = [&, kernel, arg_shape, result_shape, arg_buffer_index, out_buffer_index](   \
            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {                                   \
            kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   ectx->arena);                                                                   \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    if (reduction_axes.size() == 1 && is_optimized_et(args[0].get_element_type()))                 \
    {                                                                                              \
        if (*reduction_axes.begin() == arg_rank - 1)                                               \
        {                                                                                          \
            std::function<decltype(runtime::cpu::kernel::reduce_##K##_innermost_1rd<float, 2>)>    \
                kernel;                                                                            \
            SELECT_ETS_AND_RANK7(kernel,                                                           \
                                 result_element_type,                                              \
                                 arg_rank,                                                         \
                                 runtime::cpu::kernel::reduce_##K##_innermost_1rd);                \
            auto functor =                                                                         \
                [&, kernel, arg_shape, result_shape, arg_buffer_index, out_buffer_index](          \
                    CPURuntimeContext* ctx, CPUExecutionContext* ectx) {                           \
                    kernel(ctx->buffer_data[arg_buffer_index],                                     \
                           ctx->buffer_data[out_buffer_index],                                     \
                           arg_shape,                                                              \
                           result_shape,                                                           \
                           ectx->arena);                                                           \
                };                                                                                 \
            functors.emplace_back(functor);                                                        \
            return;                                                                                \
        }                                                                                          \
                                                                                                   \
        std::function<decltype(runtime::cpu::kernel::reduce_##K##_1rd<float, 2>)> kernel;          \
        SELECT_ETS_AND_RANK7(                                                                      \
            kernel, result_element_type, arg_rank, runtime::cpu::kernel::reduce_##K##_1rd);        \
        auto functor = [&,                                                                         \
                        kernel,                                                                    \
                        arg_shape,                                                                 \
                        result_shape,                                                              \
                        reduction_axes,                                                            \
                        arg_buffer_index,                                                          \
                        out_buffer_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {     \
            kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   reduction_axes,                                                                 \
                   ectx->arena);                                                                   \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    if (reduction_axes.size() == 2 && arg_rank == 3 &&                                             \
        is_optimized_et(args[0].get_element_type()))                                               \
    {                                                                                              \
        std::function<decltype(runtime::cpu::kernel::reduce_##K##_3d_2rd<float>)> kernel;          \
        SELECT_ETS(kernel, result_element_type, runtime::cpu::kernel::reduce_##K##_3d_2rd);        \
        auto functor = [&,                                                                         \
                        kernel,                                                                    \
                        arg_shape,                                                                 \
                        result_shape,                                                              \
                        reduction_axes,                                                            \
                        arg_buffer_index,                                                          \
                        out_buffer_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {     \
            kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   reduction_axes,                                                                 \
                   ectx->arena);                                                                   \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    if (reduction_axes.size() == 2 && arg_rank == 4 &&                                             \
        is_optimized_et(args[0].get_element_type()))                                               \
    {                                                                                              \
        std::function<decltype(runtime::cpu::kernel::reduce_##K##_4d_2rd<float>)> kernel;          \
        SELECT_ETS(kernel, result_element_type, runtime::cpu::kernel::reduce_##K##_4d_2rd);        \
        auto functor = [&,                                                                         \
                        kernel,                                                                    \
                        arg_shape,                                                                 \
                        result_shape,                                                              \
                        reduction_axes,                                                            \
                        arg_buffer_index,                                                          \
                        out_buffer_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {     \
            kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   reduction_axes,                                                                 \
                   ectx->arena);                                                                   \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    if (reduction_axes.size() == 2 && arg_rank == 5 &&                                             \
        is_optimized_et(args[0].get_element_type()))                                               \
    {                                                                                              \
        std::function<decltype(runtime::cpu::kernel::reduce_##K##_5d_2rd<float>)> kernel;          \
        SELECT_ETS(kernel, result_element_type, runtime::cpu::kernel::reduce_##K##_5d_2rd);        \
        auto functor = [&,                                                                         \
                        kernel,                                                                    \
                        arg_shape,                                                                 \
                        result_shape,                                                              \
                        reduction_axes,                                                            \
                        arg_buffer_index,                                                          \
                        out_buffer_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {     \
            kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   reduction_axes,                                                                 \
                   ectx->arena);                                                                   \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
        return;                                                                                    \
    }                                                                                              \
                                                                                                   \
    std::function<decltype(runtime::cpu::kernel::K<float>)> ref_kernel;                            \
                                                                                                   \
    SELECT_KERNEL(ref_kernel, result_element_type, runtime::cpu::kernel::K);                       \
                                                                                                   \
    auto functor = [&,                                                                             \
                    ref_kernel,                                                                    \
                    arg_shape,                                                                     \
                    result_shape,                                                                  \
                    reduction_axes,                                                                \
                    arg_buffer_index,                                                              \
                    out_buffer_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {         \
        ref_kernel(ctx->buffer_data[arg_buffer_index],                                             \
                   ctx->buffer_data[out_buffer_index],                                             \
                   arg_shape,                                                                      \
                   result_shape,                                                                   \
                   reduction_axes,                                                                 \
                   ectx->arena);                                                                   \
    };                                                                                             \
    functors.emplace_back(functor)
