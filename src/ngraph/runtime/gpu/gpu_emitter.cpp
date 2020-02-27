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
#include "ngraph/ops.hpp"
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

function<std::string(EMIT_ARGS)> runtime::gpu::GPU_Emitter::get_emit_function(const Node& node)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {<Abs typeid>, function<std::string(EMIT_ARGS)},
// {<Acos typeid>, function<std::string(EMIT_ARGS)},
// ...
#define NGRAPH_OP(a, b, VERS)                                                                      \
    {type_index(typeid(b::a)), runtime::gpu::GPU_Emitter::emit_v##VERS##_##a},
    static const map<type_index, function<std::string(EMIT_ARGS)>> typeid_map{
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

std::string runtime::gpu::GPU_Emitter::emit_v0_Abs(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Abs>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Acos(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Acos>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Add(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Add>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_And(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::And>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ArgMax(EMIT_ARGS)
{
    cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_MAX;
    return runtime::gpu::GPU_Emitter::emit_ArgReduce(
        compiled_function, function_name, node, args, out, reduce_op);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ArgMin(EMIT_ARGS)
{
    cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_MIN;
    return runtime::gpu::GPU_Emitter::emit_ArgReduce(
        compiled_function, function_name, node, args, out, reduce_op);
}

std::string runtime::gpu::GPU_Emitter::emit_ArgReduce(EMIT_ARGS, cudnnReduceTensorOp_t reduce_op)
{
    if (out[0].get_size() == 0)
    {
        // return;
        return "";
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

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

    auto index = cudnn_emitter->build_reduce_forward(
        reduce_op, dtypes, args[0].get_shape(), axis_set, CUDNNEmitter::ReductionMode::ArgReduce);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Asin(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Asin>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Atan(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Atan>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_AvgPool(EMIT_ARGS)
{
    // assumes NC{d1,d2,...} format
    auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);
    auto& input_shape = args[0].get_shape();
    auto& result_shape = out[0].get_shape();
    auto padding_below = avg_pool->get_padding_below();
    auto padding_above = avg_pool->get_padding_above();

    size_t index = 0;

    // if 1d or has asymmetric padding, must handle pooling manually
    if (input_shape.size() == 3 || padding_below != padding_above)
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();

        index = cuda_emitter->build_avg_pool({{args[0].get_type(), out[0].get_type()}},
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
        auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

        auto cudnn_avg_type = avg_pool->get_include_padding_in_avg_computation()
                                  ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                  : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

        index = cudnn_emitter->build_pooling(cudnn_avg_type,
                                             out[0].get_element_type(),
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

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_AvgPoolBackprop(EMIT_ARGS)
{
    auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);
    auto output_shape = out[0].get_shape();
    auto delta_shape = args[0].get_shape();

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

    if (output_shape.size() >= 4)
    {
        auto cudnn_avg_type = apb->get_include_padding_in_avg_computation()
                                  ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                  : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

        auto index = cudnn_emitter->build_pooling(cudnn_avg_type,
                                                  out[0].get_element_type(),
                                                  CUDNNEmitter::Prop::Backward,
                                                  output_shape,
                                                  delta_shape,
                                                  apb->get_window_movement_strides(),
                                                  apb->get_window_shape(),
                                                  apb->get_padding_below(),
                                                  apb->get_padding_above());

        return compiled_function->add_to_runtime(index, function_name, args, out);
    }
    else
    {
        throw ngraph_error("AvgPoolBackprop currently only supports tensors of rank 4 and greater");
    }
}

template <typename T>
std::string emit_BatchNorm(EMIT_ARGS, runtime::gpu::CUDNNEmitter::Prop direction, bool save_stats)
{
    const T* batchnorm = static_cast<const T*>(node);

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

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

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchNormInference(EMIT_ARGS)
{
    return ::emit_BatchNorm<ngraph::op::BatchNormInference>(
        compiled_function, function_name, node, args, out, CUDNNEmitter::Prop::Inference, false);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchNormTraining(EMIT_ARGS)
{
    return ::emit_BatchNorm<ngraph::op::BatchNormTraining>(
        compiled_function, function_name, node, args, out, CUDNNEmitter::Prop::Forward, false);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchNormTrainingWithStats(EMIT_ARGS)
{
    return ::emit_BatchNorm<ngraph::op::gpu::BatchNormTrainingWithStats>(
        compiled_function, function_name, node, args, out, CUDNNEmitter::Prop::Forward, true);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchNormTrainingBackprop(EMIT_ARGS)
{
    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

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
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Broadcast(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();
    auto& axes = broadcast->get_broadcast_axes();

    size_t index;
    // broadcast axes is empty, do a copy
    if (axes.empty())
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                           out[0].get_size() * out[0].get_element_type().size());
    }
    else
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
        index = cuda_emitter->build_broadcast(
            {{args[0].get_type(), out[0].get_type()}}, result_shape, axes);
    }
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Ceiling(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Ceiling>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Concat(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto concat = static_cast<const ngraph::op::Concat*>(node);
    auto axis = concat->get_concatenation_axis();

    vector<NVShape> input_shapes;
    for (auto arg : args)
    {
        input_shapes.push_back(arg.get_shape());
    }

    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto index =
        cuda_emitter->build_concat(out[0].get_type(), input_shapes, axis, out[0].get_shape());

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Constant(EMIT_ARGS)
{
    return "";
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Convert(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Convert>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Convolution(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }

    auto convolution = static_cast<const ngraph::op::Convolution*>(node);

    size_t index = 0;
    if (convolution->get_padding_below().size() > 3)
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
        index = cuda_emitter->build_primitive(convolution);
    }
    else
    {
        auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
        index = cudnn_emitter->build_primitive(convolution);
    }

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ConvolutionBackpropData(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        // return;
        return "";
    }

    auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

    if (convolution->get_padding_below_forward().size() > 3)
    {
        throw runtime_error(node->get_name() + "with more than 3D is not implemented.");
    }

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t index = cudnn_emitter->build_primitive(convolution);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ConvolutionBackpropFilters(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        // return;
        return "";
    }

    auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);

    if (convolution->get_padding_below_forward().size() > 3)
    {
        throw runtime_error(node->get_name() + "with more than 3D is not implemented.");
    }

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t index = cudnn_emitter->build_primitive(convolution);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Cos(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Cos>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Cosh(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Cosh>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Divide(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Divide>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Dot(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto dot = static_cast<const ngraph::op::Dot*>(node);
    size_t reduction_axes_count = dot->get_reduction_axes_count();
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    const Shape& out_shape = out[0].get_shape();

    size_t index;
    // set output to 0 if input size is 0
    if (args[0].get_size() == 0 || args[1].get_size() == 0)
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index =
            host_emitter->build_zero_out(0, out[0].get_size() * out[0].get_element_type().size());
    }
    else
    {
        auto& cublas_emitter = compiled_function->get_primitive_emitter()->get_cublas_emitter();
        index = cublas_emitter->build_dot(out[0].get_element_type(),
                                          arg0_shape,
                                          arg1_shape,
                                          out_shape,
                                          reduction_axes_count,
                                          node);
    }

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Equal(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Equal>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Exp(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Exp>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Floor(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Floor>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GetOutputElement(EMIT_ARGS)
{
    auto get_tuple_element = static_cast<const ngraph::op::GetOutputElement*>(node);
    auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
    size_t index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                              out[0].get_size() * out[0].get_element_type().size(),
                                              0,
                                              get_tuple_element->get_n());
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Greater(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Greater>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GreaterEq(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::GreaterEq>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Less(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Less>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LessEq(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::LessEq>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Log(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Log>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LRN(EMIT_ARGS)
{
    auto lrn = static_cast<const ngraph::op::LRN*>(node);
    auto& input_shape = args[0].get_shape();

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t index = cudnn_emitter->build_lrn(out[0].get_type(),
                                            CUDNNEmitter::Prop::Forward,
                                            input_shape,
                                            lrn->get_alpha(),
                                            lrn->get_beta(),
                                            lrn->get_bias(),
                                            lrn->get_nsize());

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Max(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }

    const ngraph::op::Max* max = static_cast<const ngraph::op::Max*>(node);
    vector<element::Type> dtypes;
    dtypes.push_back(args[0].get_element_type());
    dtypes.push_back(out[0].get_element_type());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    size_t index = cuda_emitter->build_reduce<ngraph::op::Max>(
        dtypes, args[0].get_shape(), out[0].get_shape(), max->get_reduction_axes());
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Maximum(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Maximum>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_MaxPool(EMIT_ARGS)
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

    size_t index;
    // 1d max pool (NCW)
    if (input_shape.size() == 3)
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();

        index = cuda_emitter->build_primitive(max_pool);
    }
    // 2d and 3d max pool (NCHW)
    else if (input_shape.size() == 4 || input_shape.size() == 5)
    {
        auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

        index = cudnn_emitter->build_primitive(max_pool);
    }
    else
    {
        throw ngraph_error("Unsupported tensor rank encountered in " + node->description());
    }

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_MaxPoolBackprop(EMIT_ARGS)
{
    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);
    auto fp_input_shape = out[0].get_shape();
    auto fp_output_shape = args[1].get_shape();

    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();

    bool needs_fprop = (args.size() != 3);
    if (fp_input_shape.size() >= 4)
    {
        auto index = cudnn_emitter->build_pooling(CUDNN_POOLING_MAX,
                                                  out[0].get_element_type(),
                                                  CUDNNEmitter::Prop::Backward,
                                                  fp_input_shape,
                                                  fp_output_shape,
                                                  mpb->get_window_movement_strides(),
                                                  mpb->get_window_shape(),
                                                  mpb->get_padding_below(),
                                                  mpb->get_padding_above(),
                                                  needs_fprop);

        return compiled_function->add_to_runtime(index, function_name, args, out);
    }
    else
    {
        throw ngraph_error("Unsupported tensor rank encountered in " + node->description());
    }
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Min(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }

    const ngraph::op::Min* min = static_cast<const ngraph::op::Min*>(node);

    vector<element::Type> dtypes;
    dtypes.push_back(args[0].get_element_type());
    dtypes.push_back(out[0].get_element_type());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    size_t index = cuda_emitter->build_reduce<ngraph::op::Min>(
        dtypes, args[0].get_shape(), out[0].get_shape(), min->get_reduction_axes());
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Minimum(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Minimum>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Multiply(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Multiply>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Negative(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Negative>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Not(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Not>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_NotEqual(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::NotEqual>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_OneHot(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto onehot = static_cast<const ngraph::op::OneHot*>(node);
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();
    auto output_datatype_size = out[0].get_element_type().size();
    size_t idx = onehot->get_one_hot_axis();

    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto index = cuda_emitter->build_onehot({{args[0].get_type(), out[0].get_type()}},
                                            arg_shape,
                                            result_shape,
                                            idx,
                                            output_datatype_size);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Or(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Or>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Pad(EMIT_ARGS)
{
    auto pad = static_cast<const ngraph::op::Pad*>(node);
    auto input_shape = args[0].get_shape();
    auto output_shape = out[0].get_shape();
    auto padding_below = pad->get_padding_below();
    auto padding_above = pad->get_padding_above();
    auto padding_interior = pad->get_padding_interior();
    auto pad_mode = pad->get_pad_mode();

    if (pad_mode != op::PadMode::CONSTANT)
    {
        throw unsupported_op("Pad modes other than CONSTANT are unsupported");
    }

    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();

    NVShape converted_padding(padding_below.begin(), padding_below.end());

    auto index =
        cuda_emitter->build_pad_fill({{args[0].get_type(), args[1].get_type(), out[0].get_type()}},
                                     input_shape,
                                     output_shape,
                                     converted_padding,
                                     padding_interior);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Parameter(EMIT_ARGS)
{
    return "";
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Power(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Power>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Product(EMIT_ARGS)
{
    const ngraph::op::Product* prod = static_cast<const ngraph::op::Product*>(node);

    if (out[0].get_size() == 0)
    {
        return "";
    }

    vector<element::Type> dtypes;
    dtypes.push_back(args[0].get_element_type());
    dtypes.push_back(out[0].get_element_type());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    size_t index = cuda_emitter->build_reduce<ngraph::op::Multiply>(
        dtypes, args[0].get_shape(), out[0].get_shape(), prod->get_reduction_axes());

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Relu(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Relu>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ReluBackprop(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::ReluBackprop>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ReplaceSlice(EMIT_ARGS)
{
    // assumes NC{d1,d2,...} format
    auto rep_slice = static_cast<const ngraph::op::ReplaceSlice*>(node);
    bool in_place_op = (args[0].get_name() == out[0].get_name());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto index = cuda_emitter->build_primitive(rep_slice, in_place_op);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Reshape(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto reshape = static_cast<const op::Reshape*>(node);

    if (out[0].get_name() == args[0].get_name() && out[0].get_offset() == args[0].get_offset())
    {
        return "// Logical reshape eliminated\n";
    }

    auto arg_shape = args[0].get_shape();
    auto arg_rank = arg_shape.size();
    auto result_shape = out[0].get_shape();
    auto input_order = reshape->get_input_order();
    size_t result_shape_product = shape_size(result_shape);

    // for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    if (!reshape->get_is_transpose() || result_shape_product < 2)
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        size_t index = host_emitter->build_memcpy(
            cudaMemcpyDeviceToDevice, out[0].get_size() * out[0].get_element_type().size());
        return compiled_function->add_to_runtime(index, function_name, args, out);
    }

    // combine inordered dimensons after reorder in shape, update output shape and input order
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

    // eleminate dimenson size = 1, update input order and output shape
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

    size_t index;
    // If there is no layout change, we can just copy.
    bool same_layout = is_sorted(new_input_order.begin(), new_input_order.end());
    if (same_layout)
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                           out[0].get_size() * out[0].get_element_type().size());
    }
    // If there *is* a layout change in the 2D case, we transpose the input.
    else
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
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
    }

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Result(EMIT_ARGS)
{
    if (args[0].get_name() == out[0].get_name())
    {
        return "// Skipping generation for " + node->get_name() + "\n";
    }

    auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
    size_t index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                              out[0].get_size() * out[0].get_element_type().size());
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Reverse(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
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
    size_t index;
    if (out[0].get_size() == 1)
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                           out[0].get_size() * out[0].get_element_type().size());
    }
    else
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
        index = cuda_emitter->build_reverse(
            {{args[0].get_type(), out[0].get_type()}}, arg_shape, reverse_axes_flag);
    }
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ReverseSequence(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto rs = static_cast<const ngraph::op::ReverseSequence*>(node);

    size_t bi = rs->get_batch_axis();
    size_t si = rs->get_sequence_axis();
    auto arg_shape0 = args[0].get_shape();
    auto arg_shape1 = args[1].get_shape();
    auto out_shape = out[0].get_shape();

    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto index = cuda_emitter->build_reverse_sequence(
        {{args[0].get_type(), args[1].get_type(), out[0].get_type()}},
        arg_shape0,
        arg_shape1,
        out_shape,
        bi,
        si);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

#if CUDNN_VERSION >= 7200
std::string runtime::gpu::GPU_Emitter::emit_v0_Rnn(EMIT_ARGS)
{
    auto rnn = static_cast<const ngraph::op::gpu::Rnn*>(node);
    auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
    size_t index = cudnn_emitter->build_primitive(rnn);
    return compiled_function->add_to_runtime(index, function_name, args, out);
}
#endif

std::string runtime::gpu::GPU_Emitter::emit_v0_Select(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Select>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sigmoid(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Sigmoid>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_SigmoidBackprop(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::SigmoidBackprop>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sign(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Sign>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sin(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Sin>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sinh(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Sinh>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Slice(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto slice = static_cast<const op::Slice*>(node);

    const auto arg_shape = args[0].get_shape();
    const auto result_shape = out[0].get_shape();
    const Coordinate& lower_bounds = slice->get_lower_bounds();
    const Strides slice_strides = slice->get_strides();

    size_t index;
    if (args[0].get_size() == out[0].get_size())
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                           out[0].get_size() * out[0].get_element_type().size());
    }
    else
    {
        auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
        index = cuda_emitter->build_slice({{args[0].get_type(), out[0].get_type()}},
                                          arg_shape,
                                          lower_bounds,
                                          slice_strides,
                                          result_shape);
    }
    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Softmax(EMIT_ARGS)
{
    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

    auto axes_set = softmax->get_axes();
    std::vector<element::Type> dtypes;
    dtypes.push_back(args[0].get_element_type());
    dtypes.push_back(out[0].get_element_type());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    size_t index = cuda_emitter->build_softmax(dtypes, args[0].get_shape(), axes_set);

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sqrt(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Sqrt>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Subtract(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Subtract>(
        compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Sum(EMIT_ARGS)
{
    return runtime::gpu::GPU_Emitter::emit_Sum_0(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_Sum_0(EMIT_ARGS)
// emit_Sum_0 uses native cuda kernels to perform Sum reduction. This method
// is faster than cudnn implementation but in its current state is less precise
// than cudnn reduce. That is causing tensorflow tests aimed at testing stabilty
// to fail
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);

    auto axes_set = sum->get_reduction_axes();
    vector<element::Type> dtypes;
    dtypes.push_back(args[0].get_element_type());
    dtypes.push_back(out[0].get_element_type());
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto sum_index = cuda_emitter->build_reduce<ngraph::op::Add>(
        dtypes, args[0].get_shape(), out[0].get_shape(), axes_set);

    return compiled_function->add_to_runtime(sum_index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_Sum_1(EMIT_ARGS)
// emit_Sum_1 uses cudnn to perform Sum reduction. This method, although
// slower than the native cuda implementation is more precise and fixes the issue with
// tensorflow test failures
{
    const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
    std::vector<element::Type> dtypes{args[0].get_element_type(), out[0].get_element_type()};
    cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_ADD;
    if (out[0].get_size() == 0)
    {
        return "";
    }
    size_t index;
    // one of args[] axes has zero size, zero output
    if (args[0].get_size() == 0)
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index =
            host_emitter->build_zero_out(0, out[0].get_size() * out[0].get_element_type().size());
    }
    else if (args[0].get_size() == out[0].get_size())
    {
        auto& host_emitter = compiled_function->get_primitive_emitter()->get_host_emitter();
        index = host_emitter->build_memcpy(cudaMemcpyDeviceToDevice,
                                           out[0].get_size() * out[0].get_element_type().size());
    }
    else
    {
        auto& cudnn_emitter = compiled_function->get_primitive_emitter()->get_cudnn_emitter();
        index = cudnn_emitter->build_reduce_forward(reduce_op,
                                                    dtypes,
                                                    args[0].get_shape(),
                                                    sum->get_reduction_axes(),
                                                    CUDNNEmitter::ReductionMode::Reduce);
    }

    return compiled_function->add_to_runtime(index, function_name, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Tan(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Tan>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Tanh(EMIT_ARGS)
{
    return emit_elementwise<ngraph::op::Tanh>(compiled_function, function_name, node, args, out);
}

std::string runtime::gpu::GPU_Emitter::emit_v0_TopK(EMIT_ARGS)
{
    if (out[0].get_size() == 0)
    {
        return "";
    }
    auto topk = static_cast<const ngraph::op::TopK*>(node);
    size_t topk_axis = topk->get_top_k_axis();
    size_t topk_k = topk->get_k();
    auto index_elem_type = topk->get_index_element_type();
    bool compute_max = topk->get_compute_max();
    std::vector<element::Type> dtypes{args[0].get_element_type()};
    NGRAPH_CHECK(out.size() == 2, "TopK can only have 2 outputs");
    for (size_t i = 0; i < out.size(); i++)
    {
        dtypes.push_back(out[i].get_element_type());
    }
    auto& input_shape = args[0].get_shape();
    auto& cuda_emitter = compiled_function->get_primitive_emitter()->get_cuda_emitter();
    auto index = cuda_emitter->build_topk(
        dtypes, input_shape, topk_axis, topk_k, index_elem_type, compute_max);

    return compiled_function->add_to_runtime(index, function_name, args, out);
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

/// List of all unsupported ops on the NVIDIA GPU backend.
std::string runtime::gpu::GPU_Emitter::emit_v0_All(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_AllReduce(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Any(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchMatMul(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BroadcastLike(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BroadcastDistributed(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ConvertLike(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Dequantize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DynReplaceSlice(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DynReshape(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DynSlice(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_EmbeddingLookup(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Erf(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Gather(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GatherND(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GenerateMask(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Passthrough(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Quantize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedConvolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedConvolutionBias(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedConvolutionBiasAdd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedConvolutionBiasSignedAdd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedConvolutionRelu(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedDot(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_QuantizedDotBias(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Recv(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Range(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ScalarConstantLike(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ScatterAdd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ScatterNDAdd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Send(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ShapeOf(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_StopGradient(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Xor(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DynBroadcast(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DynPad(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Tile(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Split(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Gelu(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Power(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Multiply(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Reverse(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CropAndResize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PriorBoxClustered(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceProd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_BinaryConvolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_NotEqual(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Greater(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_RegionYolo(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GRN(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Divide(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GroupConvolutionBackpropData(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_GreaterEqual(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Clamp(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_RandomUniform(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_LessEqual(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_GroupConvolutionBackpropData(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PriorBox(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Equal(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Gather(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_AvgPoolBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Round(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ShuffleChannels(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Transpose(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Elu(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GeluBackpropFactor(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_SquaredDifference(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ConvolutionBiasBackpropFiltersBias(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GRUCell(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_FakeQuantize(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Less(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ConvolutionBackpropData(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_DeformablePSROIPooling(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Unsqueeze(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_NormalizeL2(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_DeformableConvolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PSROIPooling(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Add(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_FloorMod(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CumSum(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Split(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ConvolutionBiasAdd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DetectionOutput(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_NonMaxSuppression(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceSum(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Maximum(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Minimum(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Squeeze(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CrossEntropyBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CrossEntropy(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Proposal(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Selu(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ConvolutionBias(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_SpaceToBatch(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_SpaceToDepth(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Select(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceMax(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_MaxPoolBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Stack(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_GatherTree(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_AvgPool(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CompiledKernel(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_RNNCell(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_BatchMatMulTranspose(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_CTCGreedyDecoder(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_BatchToSpace(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_DepthToSpace(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_HardSigmoid(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_TensorIterator(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ReorgYolo(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Atan2(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Broadcast(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceLogicalOr(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_LogicalNot(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_LogicalXor(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LSTMSequence(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ROIPooling(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceLogicalAnd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Interpolate(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LayerNorm(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_GenerateMask(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PartialSliceBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ConvolutionBackpropFilters(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Subtract(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Reshape(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PRelu(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LSTMCell(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ScaleShift(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GroupConvolutionBackpropFilters(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceMin(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_MatMul(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_MaxPool(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_Gemm(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_SoftmaxCrossEntropyBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Pad(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_ReduceMean(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Softmax(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Convolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_PartialSlice(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_StridedSlice(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_MVN(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_LogicalOr(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_Mod(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_VariadicSplit(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_ScatterND(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_GroupConvolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_SoftmaxCrossEntropy(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_LogicalAnd(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_TopK(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v1_GroupConvolution(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}

std::string runtime::gpu::GPU_Emitter::emit_v0_LayerNormBackprop(EMIT_ARGS)
{
    throw unsupported_op("Unsupported op '" + node->description() + "'");
}
