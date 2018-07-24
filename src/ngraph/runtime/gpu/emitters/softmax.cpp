/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/gpu/emitters/softmax.hpp"

#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_shape.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace ngraph;

std::vector<Shape> runtime::gpu::Emitter<op::Softmax>::get_workspaces()
{
    auto input_shape = m_node->get_inputs().at(0).get_shape();
    auto axes = m_node->get_axes();
    if (axes.size() != input_shape.size())
    {
        auto reduced_shape = input_shape;
        for (auto const& axis : axes)
        {
            reduced_shape[axis] = 1;
        }
        size_t reduced_size = shape_size(reduced_shape);

        return std::vector<Shape>{
            Shape{reduced_size * m_node->get_outputs().at(0).get_element_type().size()}};
    }
    else
    {
        return std::vector<Shape>{};
    }
}

std::vector<std::vector<int>> runtime::gpu::Emitter<op::Softmax>::get_constants()
{
    auto input_shape = m_node->get_inputs().at(0).get_shape();
    auto output_shape = m_node->get_outputs().at(0).get_shape();
    auto axes = m_node->get_axes();
    if (axes.size() != input_shape.size())
    {
        std::vector<std::vector<int>> constants;
        for (auto& tensor_shape : {input_shape, output_shape})
        {
            // calculate strides
            GPUShape strides = row_major_strides(tensor_shape);
            // precacluate invariants for integer division via multiplication
            std::vector<int> stride_magic;
            std::vector<int> stride_shift;
            for (int i = 0; i < strides.size(); i++)
            {
                int magic;
                int shift;
                std::tie(magic, shift) = idiv_magic_u64(strides[i]);
                stride_magic.push_back(magic);
                stride_shift.push_back(shift);
            }
            // calculate reduced tensor strides with 0s inserted for reduced axes
            GPUShape reduced_shape = tensor_shape;
            for (auto const& axis : axes)
            {
                reduced_shape[axis] = 1;
            }
            GPUShape reduced_strides = row_major_strides(reduced_shape);
            for (auto const& axis : axes)
            {
                reduced_strides[axis] = 0;
            }
            constants.push_back(strides);
            constants.push_back(stride_magic);
            constants.push_back(stride_shift);
            constants.push_back(reduced_strides);
        }
        return constants;
    }
    else
    {
        return std::vector<std::vector<int>>{};
    }
}

void runtime::gpu::Emitter<op::Softmax>::emit(GPU_ExternalFunction* external_function,
                                              codegen::CodeWriter& writer,
                                              const std::vector<GPU_TensorViewWrapper>& args,
                                              const std::vector<GPU_TensorViewWrapper>& out)
{
    writer.block_begin();
    {
        auto tensor_shape = args[0].get_shape();
        auto axes = m_node->get_axes();

        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

        if (axes.size() != tensor_shape.size())
        {
            auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

            // exponentiate with fused sum reduction to calculate softmax denominator
            size_t exp_sum_reduce =
                cuda_emitter->build_elementwise_collective<ngraph::op::Exp, ngraph::op::Add>(
                    {{args[0].get_type(), out[0].get_type()}},
                    args[0].get_shape(),
                    {},
                    axes,
                    true /* multi-output */);

            // ensure workspace is zeroed out
            gpu::kernel::emit_memset(
                writer, out[1], 0, out[1].get_element_type().size() * out[1].get_size());

            writer << "gpu::invoke_primitive(ctx, " << exp_sum_reduce << ", ";
            writer << "std::vector<void*>{" << args[0].get_name();
            for (auto i = 1u; i < 5; i++)
            {
                writer << ", " << args[i].get_name();
            }
            writer << "}.data(), ";
            // cache the elementwise result and the fused result (multi-output)
            writer << "std::vector<void*>{" << out[1].get_name();
            writer << ", " << out[0].get_name() << "}.data()";
            writer << ");\n";

            // inplace binary division with fused broadcast to calculate softmax
            size_t div_broadcast = cuda_emitter->build_elementwise_collective<ngraph::op::Divide>(
                {{out[0].get_type(), out[0].get_type(), out[0].get_type()}},
                out[0].get_shape(),
                {1},
                axes);

            writer << "gpu::invoke_primitive(ctx, " << div_broadcast << ", ";
            writer << "std::vector<void*>{" << out[0].get_name() << ", " << out[1].get_name();
            for (auto i = 5u; i < args.size(); i++)
            {
                writer << ", " << args[i].get_name();
            }
            writer << "}.data(), ";
            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
            writer << ");\n";
        }
        else
        {
            size_t softmax_index = cudnn_emitter->build_softmax(CUDNN_SOFTMAX_FAST,
                                                                CUDNN_SOFTMAX_MODE_INSTANCE,
                                                                out[0].get_type(),
                                                                CUDNNEmitter::Prop::Forward,
                                                                tensor_shape);
            writer << "gpu::invoke_primitive(ctx, " << softmax_index << ", ";
            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
            writer << ");\n";
        }
    }
    writer.block_end();
}
