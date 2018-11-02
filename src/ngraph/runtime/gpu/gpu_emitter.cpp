//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <nvrtc.h>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/runtime/gpu/op/rnn.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

function<void(EMIT_ARGS)> runtime::gpu::GPU_Emitter::get_emit_function(const Node& node)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {<Abs typeid>, function<void(EMIT_ARGS)},
// {<Acos typeid>, function<void(EMIT_ARGS)},
// ...
#define NGRAPH_OP(a, b) {type_index(typeid(b::a)), runtime::gpu::GPU_Emitter::emit_##a},
    static const map<type_index, function<void(EMIT_ARGS)>> typeid_map{
#include "ngraph/runtime/gpu/op/op_tbl.hpp"
    };
#undef NGRAPH_OP
    auto it = typeid_map.find(type_index(typeid(node)));
    if (it == typeid_map.end())
    {
        throw unsupported_op("Unsupported op '" + node.description() + "'");
    }

    return it->second;
}

void runtime::gpu::GPU_Emitter::emit_Abs(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Abs>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Acos(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Acos>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Add(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Add>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_AllReduce(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

void runtime::gpu::GPU_Emitter::emit_And(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::And>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_ArgMax(EMIT_ARGS)
{
    cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_MAX;
    runtime::gpu::GPU_Emitter::emit_ArgReduce(
        external_function, writer, node, args, out, reduce_op);
}

void runtime::gpu::GPU_Emitter::emit_ArgMin(EMIT_ARGS)
{
    cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_MIN;
    runtime::gpu::GPU_Emitter::emit_ArgReduce(
        external_function, writer, node, args, out, reduce_op);
}

void runtime::gpu::GPU_Emitter::emit_ArgReduce(EMIT_ARGS, cudnnReduceTensorOp_t reduce_op)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    size_t axis;
    if (reduce_op == CUDNN_REDUCE_TENSOR_MIN)
    {
        auto argmin = static_cast<const ngraph::op::ArgMin*>(node);
        axis = argmin->get_reduction_axis();
    }
    else if (reduce_op == CUDNN_REDUCE_TENSOR_MAX)
    {
        auto argmax = static_cast<const ngraph::op::ArgMax*>(node);
        axis = argmax->get_reduction_axis();
    }
    else
    {
        throw std::runtime_error("Not supported. Only Min/Max op are supported by ArgReduce.");
    }
    auto axis_set = AxisSet{axis};

    std::vector<element::Type> dtypes{args[0].get_element_type(), out[0].get_element_type()};

    writer.block_begin();
    {
        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

        auto index = cudnn_emitter->build_reduce_forward(reduce_op,
                                                         dtypes,
                                                         args[0].get_shape(),
                                                         axis_set,
                                                         CUDNNEmitter::ReductionMode::ArgReduce);

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Asin(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Asin>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Atan(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Atan>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_AvgPool(EMIT_ARGS)
{
    // assumes NC{d1,d2,...} format
    auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);
    writer.block_begin();
    {
        auto& input_shape = args[0].get_shape();
        auto& result_shape = out[0].get_shape();
        auto padding_below = avg_pool->get_padding_below();
        auto padding_above = avg_pool->get_padding_above();

        size_t index = 0;

        // if 1d or has asymmetric padding, must handle pooling manually
        if (input_shape.size() == 3 || padding_below != padding_above)
        {
            auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

            index =
                cuda_emitter->build_avg_pool({{args[0].get_type(), out[0].get_type()}},
                                             input_shape,
                                             result_shape,
                                             avg_pool->get_window_shape(),
                                             avg_pool->get_window_movement_strides(),
                                             padding_below,
                                             avg_pool->get_include_padding_in_avg_computation());
        }
        // 2d and 3d avg pool (NCHW) with either symetric padding or no padding
        else if (input_shape.size() == 4 || input_shape.size() == 5)
        {
            auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

            auto cudnn_avg_type = avg_pool->get_include_padding_in_avg_computation()
                                      ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                      : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

            index = cudnn_emitter->build_pooling(cudnn_avg_type,
                                                 out[0].get_type(),
                                                 CUDNNEmitter::Prop::Forward,
                                                 input_shape,
                                                 result_shape,
                                                 avg_pool->get_window_movement_strides(),
                                                 avg_pool->get_window_shape(),
                                                 padding_below,
                                                 padding_above);
        }
        else
        {
            throw runtime_error("Pooling currently only supports up to 3 spatial dimensions.");
        }

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_AvgPoolBackprop(EMIT_ARGS)
{
    writer.block_begin();
    {
        auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);
        auto output_shape = out[0].get_shape();
        auto delta_shape = args[0].get_shape();

        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

        if (output_shape.size() >= 4)
        {
            auto cudnn_avg_type = apb->get_include_padding_in_avg_computation()
                                      ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                      : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

            auto index = cudnn_emitter->build_pooling(cudnn_avg_type,
                                                      out[0].get_type(),
                                                      CUDNNEmitter::Prop::Backward,
                                                      output_shape,
                                                      delta_shape,
                                                      apb->get_window_movement_strides(),
                                                      apb->get_window_shape(),
                                                      apb->get_padding_below(),
                                                      apb->get_padding_above());

            // cuDNN backwards pooling requests input and output tensors from
            // the forward pass but does not use them. It also behaves differently
            // for max pool vs avg pool. The repetition of args below is to address
            // this interface in a way that supports both max and avg pooling
            writer << "void* input[] = {" << node_names(args, {0, 0}) << "};\n";
            writer << "void* output[] = {" << node_names(out) << "};\n";
            writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
        }
    }
    writer.block_end();
}

template <typename T>
void emit_BatchNorm(EMIT_ARGS, runtime::gpu::CUDNNEmitter::Prop direction, bool save_stats)
{
    const T* batchnorm = static_cast<const T*>(node);

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

    bool global_stats = false;
    if (direction == runtime::gpu::CUDNNEmitter::Prop::Forward)
    {
        global_stats = (batchnorm->get_arguments().size() == 5);
    }

    auto index = cudnn_emitter->build_batchnorm(CUDNN_BATCHNORM_SPATIAL,
                                                out[0].get_type(),
                                                direction,
                                                args[2].get_shape(),
                                                args[0].get_shape(),
                                                batchnorm->get_eps_value(),
                                                global_stats,
                                                save_stats);

    writer.block_begin();
    {
        writer << "void* input[] = {" << runtime::gpu::GPU_Emitter::node_names(args) << "};\n";
        writer << "void* output[] = {" << runtime::gpu::GPU_Emitter::node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_BatchNormInference(EMIT_ARGS)
{
    ::emit_BatchNorm<ngraph::op::BatchNormInference>(
        external_function, writer, node, args, out, CUDNNEmitter::Prop::Inference, false);
}

void runtime::gpu::GPU_Emitter::emit_BatchNormTraining(EMIT_ARGS)
{
    ::emit_BatchNorm<ngraph::op::BatchNormTraining>(
        external_function, writer, node, args, out, CUDNNEmitter::Prop::Forward, false);
}

void runtime::gpu::GPU_Emitter::emit_BatchNormTrainingWithStats(EMIT_ARGS)
{
    ::emit_BatchNorm<ngraph::op::gpu::BatchNormTrainingWithStats>(
        external_function, writer, node, args, out, CUDNNEmitter::Prop::Forward, true);
}

void runtime::gpu::GPU_Emitter::emit_BatchNormTrainingBackprop(EMIT_ARGS)
{
    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

    bool needs_variance_inversion = false;
    auto annotation = batchnorm->get_op_annotations();
    if (annotation)
    {
        auto bnbp_annotation =
            std::dynamic_pointer_cast<runtime::gpu::BatchNormBackpropAnnotations>(annotation);
        if (bnbp_annotation && bnbp_annotation->has_inverted_variance() == false)
        {
            needs_variance_inversion = true;
        }
    }
    auto index = cudnn_emitter->build_batchnorm(CUDNN_BATCHNORM_SPATIAL,
                                                out[0].get_type(),
                                                CUDNNEmitter::Prop::Backward,
                                                args[2].get_shape(),
                                                args[0].get_shape(),
                                                batchnorm->get_eps_value(),
                                                false,
                                                false,
                                                needs_variance_inversion);
    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Broadcast(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    auto& axes = broadcast->get_broadcast_axes();
    // broadcast axes is empty, do a copy
    if (axes.empty())
    {
        writer.block_begin();
        kernel::emit_memcpyDtD(writer, out[0], args[0]);
        writer.block_end();
        return;
    }

    auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

    auto bcast_index = cuda_emitter->build_broadcast(
        {{args[0].get_type(), out[0].get_type()}}, result_shape, axes);
    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << bcast_index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Ceiling(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Ceiling>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Concat(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto concat = static_cast<const ngraph::op::Concat*>(node);
    auto axis = concat->get_concatenation_axis();

    vector<NVShape> input_shapes;
    for (auto arg : args)
    {
        input_shapes.push_back(arg.get_shape());
    }

    auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
    auto index =
        cuda_emitter->build_concat(out[0].get_type(), input_shapes, axis, out[0].get_shape());

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Constant(EMIT_ARGS)
{
}

void runtime::gpu::GPU_Emitter::emit_Convert(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Convert>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Convolution(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    auto convolution = static_cast<const ngraph::op::Convolution*>(node);

    size_t conv_index = 0;
    if (convolution->get_padding_below().size() > 3)
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
        conv_index = cuda_emitter->build_primitive(convolution);
    }
    else
    {
        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
        conv_index = cudnn_emitter->build_primitive(convolution);
    }

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << conv_index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_ConvolutionBackpropData(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

    if (convolution->get_padding_below_forward().size() > 3)
    {
        throw runtime_error(node->get_name() + "with more than 3D is not implemented.");
    }

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t conv_index = cudnn_emitter->build_primitive(convolution);

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << conv_index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_ConvolutionBackpropFilters(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);

    if (convolution->get_padding_below_forward().size() > 3)
    {
        throw runtime_error(node->get_name() + "with more than 3D is not implemented.");
    }

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t conv_index = cudnn_emitter->build_primitive(convolution);

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << conv_index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Cos(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Cos>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Cosh(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Cosh>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Divide(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Divide>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Dequantize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

void runtime::gpu::GPU_Emitter::emit_Dot(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto dot = static_cast<const ngraph::op::Dot*>(node);
    size_t reduction_axes_count = dot->get_reduction_axes_count();
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    const Shape& out_shape = out[0].get_shape();

    writer.block_begin();
    {
        // set output to 0 if input size is 0
        if (args[0].get_size() == 0 || args[1].get_size() == 0)
        {
            writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, "
                   << out[0].get_size() << " * " << out[0].get_element_type().size() << ");\n";
        }

        else
        {
            auto& cublas_emitter = external_function->get_primitive_emitter()->get_cublas_emitter();

            auto index = cublas_emitter->build_dot(out[0].get_element_type(),
                                                   arg0_shape,
                                                   arg1_shape,
                                                   out_shape,
                                                   reduction_axes_count,
                                                   node);

            writer << "void* input[] = {" << node_names(args) << "};\n";
            writer << "void* output[] = {" << node_names(out) << "};\n";
            writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Equal(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Equal>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Exp(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Exp>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Floor(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Floor>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_FunctionCall(EMIT_ARGS)
{
    auto function_call = static_cast<const ngraph::op::FunctionCall*>(node);
    shared_ptr<Function> function = function_call->get_functions()[0];

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << function->get_name() << "(input, output, ctx);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_GenerateMask(EMIT_ARGS)
{
    throw ngraph_error("GenerateMask is not supported yet on NVIDIA GPU");
}

void runtime::gpu::GPU_Emitter::emit_GetOutputElement(EMIT_ARGS)
{
    auto get_tuple_element = static_cast<const ngraph::op::GetOutputElement*>(node);

    writer.block_begin();
    writer << "runtime::gpu::cuda_memcpyDtD(" << out[0].get_name() << ", "
           << args[get_tuple_element->get_n()].get_name() << ", "
           << out[0].get_size() * out[0].get_element_type().size() << ");\n";
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Greater(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Greater>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_GreaterEq(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::GreaterEq>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Less(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Less>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_LessEq(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::LessEq>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Log(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Log>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_LRN(EMIT_ARGS)
{
    auto lrn = static_cast<const ngraph::op::LRN*>(node);
    auto& input_shape = args[0].get_shape();

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

    size_t index = cudnn_emitter->build_lrn(out[0].get_type(),
                                            CUDNNEmitter::Prop::Forward,
                                            input_shape,
                                            lrn->get_alpha(),
                                            lrn->get_beta(),
                                            lrn->get_bias(),
                                            lrn->get_nsize());
    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Max(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    const ngraph::op::Max* max = static_cast<const ngraph::op::Max*>(node);
    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
    auto index = cudnn_emitter->build_primitive(max);

    writer.block_begin();
    writer << "void* input[] = {" << node_names(args) << "};\n";
    writer << "void* output[] = {" << node_names(out) << "};\n";
    writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Maximum(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Maximum>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_MaxPool(EMIT_ARGS)
{
    // assumes NC{d1,d2,...} format
    auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

    auto& input_shape = args[0].get_shape();
    auto padding_below = max_pool->get_padding_below();
    auto padding_above = max_pool->get_padding_above();
    if (input_shape.size() < 3)
    {
        throw runtime_error(
            "MaxPool operation requested for a tensor of less than 3 dimensions. "
            "Tensors should have at least one spatial dimension, dim(NC{d1...dN}) "
            "<= 3");
    }
    else if (input_shape.size() > 5)
    {
        throw runtime_error("Pooling currently only supports up to 3 spatial dimensions.");
    }

    size_t max_pool_index;
    // 1d max pool (NCW)
    if (input_shape.size() == 3)
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

        max_pool_index = cuda_emitter->build_primitive(max_pool);
    }
    // 2d and 3d max pool (NCHW)
    else if (input_shape.size() == 4 || input_shape.size() == 5)
    {
        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

        max_pool_index = cudnn_emitter->build_primitive(max_pool);
    }
    writer.block_begin();
    writer << "void* input[] = {" << node_names(args) << "};\n";
    writer << "void* output[] = {" << node_names(out) << "};\n";
    writer << "gpu::invoke_primitive(ctx, " << max_pool_index << ", input, output);\n";
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_MaxPoolBackprop(EMIT_ARGS)
{
    writer.block_begin();
    {
        auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);
        auto fp_input_shape = out[0].get_shape();
        auto fp_output_shape = args[1].get_shape();

        auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

        if (fp_input_shape.size() >= 4)
        {
            auto index = cudnn_emitter->build_pooling(CUDNN_POOLING_MAX,
                                                      out[0].get_type(),
                                                      CUDNNEmitter::Prop::Backward,
                                                      fp_input_shape,
                                                      fp_output_shape,
                                                      mpb->get_window_movement_strides(),
                                                      mpb->get_window_shape(),
                                                      mpb->get_padding_below(),
                                                      mpb->get_padding_above());

            writer << "void* input[] = {" << node_names(args) << "};\n";
            writer << "void* output[] = {" << node_names(out) << "};\n";
            writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Min(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    const ngraph::op::Min* min = static_cast<const ngraph::op::Min*>(node);
    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
    auto index = cudnn_emitter->build_primitive(min);

    writer.block_begin();
    writer << "void* input[] = {" << node_names(args) << "};\n";
    writer << "void* output[] = {" << node_names(out) << "};\n";
    writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Minimum(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Minimum>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Multiply(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Multiply>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Negative(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Negative>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Not(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Not>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_NotEqual(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::NotEqual>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_OneHot(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto onehot = static_cast<const ngraph::op::OneHot*>(node);
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();
    auto output_datatype_size = out[0].get_element_type().size();
    size_t idx = onehot->get_one_hot_axis();

    writer.block_begin();
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
        auto index = cuda_emitter->build_onehot({{args[0].get_type(), out[0].get_type()}},
                                                arg_shape,
                                                result_shape,
                                                idx,
                                                output_datatype_size);

        writer.block_begin();
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
        writer.block_end();
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Or(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Or>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Pad(EMIT_ARGS)
{
    auto pad = static_cast<const ngraph::op::Pad*>(node);
    writer.block_begin();
    {
        auto input_shape = args[0].get_shape();
        auto output_shape = out[0].get_shape();
        auto padding_below = pad->get_padding_below();
        auto padding_above = pad->get_padding_above();
        auto padding_interior = pad->get_padding_interior();

        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

        auto pad_index = cuda_emitter->build_pad_fill(
            {{args[0].get_type(), args[1].get_type(), out[0].get_type()}},
            input_shape,
            output_shape,
            padding_below,
            padding_interior);
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << pad_index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Parameter(EMIT_ARGS)
{
}

void runtime::gpu::GPU_Emitter::emit_Power(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Power>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Product(EMIT_ARGS)
{
    const ngraph::op::Product* product = static_cast<const ngraph::op::Product*>(node);

    writer.block_begin();
    {
        if (out[0].get_size() != 0)
        {
            // one of args[] axes has zero size, fill output with 1
            if (args[0].get_size() == 0)
            {
                writer << out[0].get_type() << " init_value = 1;\n";
                writer << "vector<" << out[0].get_type() << "> temp(" << out[0].get_size()
                       << ", init_value);\n";
                writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                       << ", (void*)temp.data(), " << out[0].get_size() << " * "
                       << out[0].get_element_type().size() << ");\n";
            }
            else if (args[0].get_size() == out[0].get_size())
            {
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
            }
            // descriptors for tensors  with <= 4 dimensions
            else
            {
                std::vector<element::Type> dtypes{args[0].get_element_type(),
                                                  out[0].get_element_type()};
                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                auto index =
                    cudnn_emitter->build_reduce_forward(CUDNN_REDUCE_TENSOR_MUL,
                                                        dtypes,
                                                        args[0].get_shape(),
                                                        product->get_reduction_axes(),
                                                        CUDNNEmitter::ReductionMode::Reduce);

                writer << "void* input[] = {" << node_names(args) << "};\n";
                writer << "void* output[] = {" << node_names(out) << "};\n";
                writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
            }
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Quantize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

void runtime::gpu::GPU_Emitter::emit_Reduce(EMIT_ARGS)
{
    // reduction function supported by GPU
    // CUDNN_REDUCE_TENSOR_ADD
    // CUDNN_REDUCE_TENSOR_MUL
    // CUDNN_REDUCE_TENSOR_MIN
    // CUDNN_REDUCE_TENSOR_MAX
    // CUDNN_REDUCE_TENSOR_AMAX
    // CUDNN_REDUCE_TENSOR_AVG
    // CUDNN_REDUCE_TENSOR_NORM1
    // CUDNN_REDUCE_TENSOR_NORM2
    // CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS

    static const unordered_map<type_index, cudnnReduceTensorOp_t> reduce_map{
        {TI(ngraph::op::Add), CUDNN_REDUCE_TENSOR_ADD},
        {TI(ngraph::op::Multiply), CUDNN_REDUCE_TENSOR_MUL},
        {TI(ngraph::op::Maximum), CUDNN_REDUCE_TENSOR_MAX},
        {TI(ngraph::op::Minimum), CUDNN_REDUCE_TENSOR_MIN}};
    const ngraph::op::Reduce* reduce_op = static_cast<const ngraph::op::Reduce*>(node);
    writer.block_begin();
    {
        if (out[0].get_size() != 0)
        {
            // one of args0 axes has zero size, zero output, use args1 value
            if (args[0].get_size() == 0)
            {
                writer << out[0].get_type() << " init_value;\n";
                writer << "runtime::gpu::cuda_memcpyDtH(&init_value, " << args[1].get_name() << " ,"
                       << args[1].get_element_type().size() << ");\n";
                writer << "vector<" << out[0].get_type() << "> temp(" << out[0].get_size()
                       << ", init_value);\n";
                writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                       << ", (void*)temp.data(), " << out[0].get_size() << " * "
                       << out[0].get_element_type().size() << ");\n";
            }
            else if (args[0].get_size() == out[0].get_size())
            {
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
            }
            else
            {
                // in current implementation:
                // 1. reduction function should only have one op
                // 2. the op should be in the op_map
                // otherwise, throw an error message
                cudnnReduceTensorOp_t reduce_tensor_op;
                auto reduction_function_ops = reduce_op->get_functions()[0]->get_ops();
                int op_count = 0;
                for (auto op : reduction_function_ops)
                {
                    if (op->is_constant() || op->is_parameter() || op->is_output())
                    {
                        continue;
                    }
                    op_count++;
                    // Work around a compiler warning (*node inside typeid may have effects
                    // with shared pointers, which is fine here but clang doesn't like it.)
                    auto& fn = *op;
                    auto f_ptr = reduce_map.find(type_index(typeid(fn)));
                    if (f_ptr == reduce_map.end())
                    {
                        throw runtime_error("reduce with function " + fn.get_name() +
                                            " is not implement yet.");
                    }
                    else if (op_count != 1)
                    {
                        throw runtime_error("reduce with more than one op is not implement yet.");
                    }
                    else
                    {
                        reduce_tensor_op = f_ptr->second;
                    }
                }
                std::vector<element::Type> dtypes{args[0].get_element_type(),
                                                  out[0].get_element_type()};
                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                auto reduce_index =
                    cudnn_emitter->build_reduce_forward(reduce_tensor_op,
                                                        dtypes,
                                                        args[0].get_shape(),
                                                        reduce_op->get_reduction_axes(),
                                                        CUDNNEmitter::ReductionMode::Reduce);

                writer << "void* input[] = {" << node_names(args) << "};\n";
                writer << "void* output[] = {" << node_names(out) << "};\n";
                writer << "gpu::invoke_primitive(ctx, " << reduce_index << ", input, output);\n";
            }
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_ReduceWindow(EMIT_ARGS)
{
    static const unordered_map<type_index, ngraph::runtime::gpu::OpName> reduce_window_map{
        {TI(ngraph::op::Add), ngraph::runtime::gpu::OpName::add},
        {TI(ngraph::op::Multiply), ngraph::runtime::gpu::OpName::multiply},
        {TI(ngraph::op::Maximum), ngraph::runtime::gpu::OpName::maximum},
        {TI(ngraph::op::Minimum), ngraph::runtime::gpu::OpName::minimum}};

    const ngraph::op::ReduceWindow* reduce_window_op =
        static_cast<const ngraph::op::ReduceWindow*>(node);
    writer.block_begin();
    {
        if (out[0].get_size() != 0)
        {
            // one of args0 axes has zero size, zero output, use args1 value
            if (args[0].get_size() == 0)
            {
                writer << out[0].get_type() << " init_value;\n";
                writer << "runtime::gpu::cuda_memcpyDtH(&init_value, " << args[1].get_name() << " ,"
                       << args[1].get_element_type().size() << ");\n";
                writer << "vector<" << out[0].get_type() << "> temp(" << out[0].get_size()
                       << ", init_value);\n";
                writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                       << ", (void*)temp.data(), " << out[0].get_size() << " * "
                       << out[0].get_element_type().size() << ");\n";
            }
            else if (args[0].get_size() == out[0].get_size())
            {
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
            }
            else
            {
                // in current implementation:
                // 1. reduction function should only have one op
                // 2. the op should be in the op_map
                // otherwise, throw an error message
                auto reduction_function_ops = reduce_window_op->get_functions()[0]->get_ops();
                unordered_map<type_index, ngraph::runtime::gpu::OpName>::const_iterator it =
                    reduce_window_map.end();
                int op_count = 0;
                for (auto op : reduction_function_ops)
                {
                    if (op->is_constant() || op->is_parameter() || op->is_output())
                    {
                        continue;
                    }
                    op_count++;
                    // Work around a compiler warning (*node inside typeid may have effects
                    // with shared pointers, which is fine here but clang doesn't like it.)
                    auto& fn = *op;
                    auto f_ptr = reduce_window_map.find(type_index(typeid(fn)));
                    if (op_count != 1)
                    {
                        throw runtime_error("reduce with more than one op is not implement yet.");
                    }
                    else if (f_ptr == reduce_window_map.end())
                    {
                        throw runtime_error("reduce with function " + fn.get_name() +
                                            " is not implement yet.");
                    }
                    else
                    {
                        it = f_ptr;
                    }
                }

                if (it == reduce_window_map.end())
                {
                    throw runtime_error("no valid op found in reduction function.");
                }

                auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
                size_t reduce_index;

                // this dtypes is two build the binary op, expect both input has same type with args[0]
                vector<string> dtypes{args[0].get_type(), args[0].get_type(), out[0].get_type()};

                reduce_index = cuda_emitter->build_reduce_window(
                    it->second,
                    dtypes,
                    args[0].get_shape(),
                    out[0].get_shape(),
                    reduce_window_op->get_window_shape(),
                    reduce_window_op->get_window_movement_strides());

                writer << "void* input[] = {" << node_names(args) << "};\n";
                writer << "void* output[] = {" << node_names(out) << "};\n";
                writer << "gpu::invoke_primitive(ctx, " << reduce_index << ", input, output);\n";
            }
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Relu(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Relu>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_ReluBackprop(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::ReluBackprop>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_ReplaceSlice(EMIT_ARGS)
{
    // assumes NC{d1,d2,...} format
    auto rep_slice = static_cast<const ngraph::op::ReplaceSlice*>(node);
    bool in_place_op = (args[0].get_name() == out[0].get_name());
    writer.block_begin();
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

        auto index = cuda_emitter->build_primitive(rep_slice, in_place_op);

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Reshape(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto reshape = static_cast<const op::Reshape*>(node);

    if (out[0].get_name() == args[0].get_name())
    {
        writer << "// Logical reshape eliminated\n";
        return;
    }

    auto arg_shape = args[0].get_shape();
    auto arg_rank = arg_shape.size();
    auto result_shape = out[0].get_shape();
    auto input_order = reshape->get_input_order();
    size_t result_shape_product = shape_size(result_shape);

    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    if (!reshape->get_is_transpose() || result_shape_product < 2)
    {
        writer.block_begin();
        {
            kernel::emit_memcpyDtD(writer, out[0], args[0]);
        }
        writer.block_end();
        return;
    }

    //combine inordered dimensons after reorder in shape, update output shape and input order
    Shape in_order_map(arg_rank, 0);
    for (int i = 0; i < arg_rank - 1; i++)
    {
        if (static_cast<int64_t>(input_order[i + 1]) - static_cast<int64_t>(input_order[i]) == 1)
        {
            in_order_map[input_order[i]] = 1;
        }
    }

    Shape combine_arg_shape;
    Shape combine_idx_map(arg_rank, 0);
    Shape combine_input_order;
    size_t shape_i = 1;
    size_t combine_rank = 0;
    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[i] == 1)
        {
            shape_i *= arg_shape[i];
        }
        else
        {
            combine_arg_shape.push_back(shape_i * arg_shape[i]);
            shape_i = 1;
            combine_idx_map[i] = combine_rank++;
        }
    }

    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[input_order[i]] == 0)
        {
            combine_input_order.push_back(combine_idx_map[input_order[i]]);
        }
    }

    //eleminate dimenson size = 1, update input order and output shape
    Shape new_arg_shape;
    Shape new_result_shape;
    Shape new_idx_map(combine_rank, 0);
    Shape new_input_order;
    size_t new_rank = 0;
    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[i] != 1)
        {
            new_arg_shape.push_back(combine_arg_shape[i]);
            new_idx_map[i] = new_rank++;
        }
    }
    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[combine_input_order[i]] != 1)
        {
            new_input_order.push_back(new_idx_map[combine_input_order[i]]);
        }
    }
    for (int i = 0; i < new_rank; i++)
    {
        new_result_shape.push_back(new_arg_shape[new_input_order[i]]);
    }

    // If there is no layout change, we can just copy.
    writer.block_begin();
    {
        bool same_layout = is_sorted(new_input_order.begin(), new_input_order.end());
        if (same_layout)
        {
            kernel::emit_memcpyDtD(writer, out[0], args[0]);
        }
        // If there *is* a layout change in the 2D case, we transpose the input.
        else
        {
            writer << "void* input[] = {" << node_names(args) << "};\n";
            writer << "void* output[] = {" << node_names(out) << "};\n";
            auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
            size_t index;
            if (new_rank == 2)
            {
                index = cuda_emitter->build_reshape_2d(
                    {{args[0].get_type(), out[0].get_type()}}, new_arg_shape, new_input_order);
            }
            // If there *is* a layout change in the 3D case, we do 3D tiled reshape.
            else if (new_rank == 3)
            {
                index = cuda_emitter->build_reshape_3d(
                    {{args[0].get_type(), out[0].get_type()}}, new_arg_shape, new_input_order);
            }
            // Other cases (reordering of axes for tensors with rank>3).
            else
            {
                index = cuda_emitter->build_reshape(
                    {{args[0].get_type(), out[0].get_type()}}, new_arg_shape, new_input_order);
            }
            writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Result(EMIT_ARGS)
{
    if (args[0].get_name() == out[0].get_name())
    {
        writer << "// Skipping generation for " << node->get_name() << "\n";
        return;
    }

    writer.block_begin();
    kernel::emit_memcpyDtD(writer, out[0], args[0]);
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Reverse(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto reverse = static_cast<const op::Reverse*>(node);

    const auto arg_shape = args[0].get_shape();
    const auto arg_rank = arg_shape.size();
    const auto result_shape = out[0].get_shape();
    const auto reverse_axes = reverse->get_reversed_axes();
    vector<uint32_t> reverse_axes_flag(arg_rank, 0);
    for (auto a : reverse_axes)
    {
        reverse_axes_flag[a] = 1;
    }
    writer.block_begin();
    if (out[0].get_size() == 1)
    {
        kernel::emit_memcpyDtD(writer, out[0], args[0]);
    }
    else
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
        auto index = cuda_emitter->build_reverse(
            {{args[0].get_type(), out[0].get_type()}}, arg_shape, reverse_axes_flag);

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_ReverseSequence(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto rs = static_cast<const ngraph::op::ReverseSequence*>(node);

    size_t bi = rs->get_batch_axis();
    size_t si = rs->get_sequence_axis();
    auto arg_shape0 = args[0].get_shape();
    auto arg_shape1 = args[1].get_shape();
    auto out_shape = out[0].get_shape();

    auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

    auto rs_index = cuda_emitter->build_reverse_sequence(
        {{args[0].get_type(), args[1].get_type(), out[0].get_type()}},
        arg_shape0,
        arg_shape1,
        out_shape,
        bi,
        si);
    writer.block_begin();
    writer << "void* input[] = {" << node_names(args) << "};\n";
    writer << "void* output[] = {" << node_names(out) << "};\n";
    writer << "gpu::invoke_primitive(ctx, " << rs_index << ", input, output);\n";
    writer.block_end();
}

#if CUDNN_VERSION >= 7200
void runtime::gpu::GPU_Emitter::emit_Rnn(EMIT_ARGS)
{
    auto rnn = static_cast<const ngraph::op::gpu::Rnn*>(node);

    auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t index = cudnn_emitter->build_primitive(rnn);

    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}
#endif

void runtime::gpu::GPU_Emitter::emit_Select(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Select>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_SelectAndScatter(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

void runtime::gpu::GPU_Emitter::emit_Sigmoid(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Sigmoid>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_SigmoidBackprop(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::SigmoidBackprop>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Sign(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Sign>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Sin(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Sin>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Sinh(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Sinh>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Slice(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto slice = static_cast<const op::Slice*>(node);

    const auto arg_shape = args[0].get_shape();
    const auto result_shape = out[0].get_shape();
    const Coordinate& lower_bounds = slice->get_lower_bounds();
    const Strides slice_strides = slice->get_strides();

    writer.block_begin();
    if (args[0].get_size() == out[0].get_size())
    {
        kernel::emit_memcpyDtD(writer, out[0], args[0]);
    }
    else
    {
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
        auto index = cuda_emitter->build_slice({{args[0].get_type(), out[0].get_type()}},
                                               arg_shape,
                                               lower_bounds,
                                               slice_strides,
                                               result_shape);

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Softmax(EMIT_ARGS)
{
    auto softmax = static_cast<const ngraph::op::Softmax*>(node);
    writer.block_begin();
    {
        size_t index;
        if (softmax->get_axes().size() != args[0].get_shape().size())
        {
            auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

            index = cuda_emitter->build_primitive(softmax);
        }
        else
        {
            auto& cudnn_emitter = external_function->get_primitive_emitter()->get_cudnn_emitter();

            index = cudnn_emitter->build_softmax(CUDNN_SOFTMAX_FAST,
                                                 CUDNN_SOFTMAX_MODE_INSTANCE,
                                                 out[0].get_type(),
                                                 CUDNNEmitter::Prop::Forward,
                                                 args[0].get_shape());
        }

        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Sqrt(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Sqrt>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_StopGradient(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

void runtime::gpu::GPU_Emitter::emit_Subtract(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Subtract>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Sum(EMIT_ARGS)
{
    const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
    writer.block_begin();
    {
        if (out[0].get_size() != 0)
        {
            // one of args[] axes has zero size, zero output
            if (args[0].get_size() == 0)
            {
                kernel::emit_memset(writer, out[0], 0);
            }
            else if (args[0].get_size() == out[0].get_size())
            {
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
            }
            else
            {
                auto axes_set = sum->get_reduction_axes();
                ngraph::AxisVector axes_vec;
                for (auto a : axes_set)
                {
                    axes_vec.push_back(a);
                }
                vector<string> dtypes;
                dtypes.push_back(args[0].get_type());
                dtypes.push_back(out[0].get_type());
                auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
                auto sum_index = cuda_emitter->build_reduce<ngraph::op::Add>(
                    dtypes, out[0].get_element_type().size(), args[0].get_shape(), axes_vec);

                writer << "void* input[] = {" << node_names(args) << "};\n";
                writer << "void* output[] = {" << node_names(out) << "};\n";
                writer << "gpu::invoke_primitive(ctx, " << sum_index << ", input, output);\n";
            }
        }
    }
    writer.block_end();
}

void runtime::gpu::GPU_Emitter::emit_Tan(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Tan>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_Tanh(EMIT_ARGS)
{
    emit_elementwise<ngraph::op::Tanh>(external_function, writer, node, args, out);
}

void runtime::gpu::GPU_Emitter::emit_TopK(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto topk = static_cast<const ngraph::op::TopK*>(node);
    size_t topk_axis = topk->get_top_k_axis();
    size_t topk_k = topk->get_k();
    auto index_elem_type = topk->get_index_element_type();
    bool compute_max = topk->get_compute_max();
    std::vector<element::Type> dtypes{args[0].get_element_type()};
    NGRAPH_ASSERT(out.size() == 2) << "TopK can only have 2 outputs";
    for (size_t i = 0; i < out.size(); i++)
    {
        dtypes.push_back(out[i].get_element_type());
    }
    auto& input_shape = args[0].get_shape();
    auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();
    auto index = cuda_emitter->build_topk(
        dtypes, input_shape, topk_axis, topk_k, index_elem_type, compute_max);
    writer.block_begin();
    {
        writer << "void* input[] = {" << node_names(args) << "};\n";
        writer << "void* output[] = {" << node_names(out) << "};\n";
        writer << "gpu::invoke_primitive(ctx, " << index << ", input, output);\n";
    }
    writer.block_end();
}

string runtime::gpu::GPU_Emitter::node_names(const vector<GPUTensorWrapper>& args,
                                             initializer_list<int> arg_indexes)
{
    vector<string> names;
    vector<int> indexes = arg_indexes;
    if (indexes.empty())
    {
        indexes = vector<int>(args.size());
        iota(indexes.begin(), indexes.end(), 0);
    }
    for (int i : indexes)
    {
        names.push_back(args[i].get_name());
    }
    return ngraph::join(names);
}

// assumes NC{d1,d2,d3,...} format
Shape runtime::gpu::get_padded_shape(const Shape& input_shape,
                                     const Shape& padding_below,
                                     const Shape& padding_above,
                                     const Shape& padding_interior)
{
    Shape padded_shape = input_shape;
    int64_t i = input_shape.size() - 1;
    int64_t j = padding_below.size() - 1;
    if (padding_interior.empty())
    {
        for (; j >= 0; j--, i--)
        {
            padded_shape[i] += padding_below[j] + padding_above[j];
        }
    }
    else
    {
        for (; j >= 0; j--, i--)
        {
            padded_shape[i] = (padded_shape[i] - 1) * padding_interior[j] + 1 + padding_below[j] +
                              padding_above[j];
        }
    }
    return padded_shape;
}
