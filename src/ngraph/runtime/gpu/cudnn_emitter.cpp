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
#include <iostream>
#include <sstream>
#include <vector>

#include "ngraph/log.hpp"
#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

cudnnTensorDescriptor_t& runtime::gpu::CUDNNEmitter::tensor_descriptor_from_shape(
    const Shape& shape, const cudnnDataType_t data_type, const cudnnTensorFormat_t tensor_format)
{
    cudnnTensorDescriptor_t& desc = m_descriptors.build<cudnnTensorDescriptor_t>();
    if (shape.size() < 4)
    {
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(shape[i]);
        }
        CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(desc,
                                                   tensor_format,
                                                   data_type,
                                                   dimensions[0],
                                                   dimensions[1],
                                                   dimensions[2],
                                                   dimensions[3]));
    }
    else if (shape.size() == 4)
    {
        CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(desc,
                                                   tensor_format,
                                                   data_type,
                                                   static_cast<int>(shape[0]),
                                                   static_cast<int>(shape[1]),
                                                   static_cast<int>(shape[2]),
                                                   static_cast<int>(shape[3])));
    }
    else
    {
        std::vector<int> dimensions(shape.size());
        for (auto i = 0u; i < shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(shape[i]);
        }
        CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            static_cast<int>(dimensions.size()),
            dimensions.data(),
            runtime::gpu::cudnn_util::compute_strides(dimensions).data()));
    }

    return desc;
}

std::vector<int> runtime::gpu::cudnn_util::compute_strides(const Shape& shape)
{
    return runtime::gpu::cudnn_util::get_vector_int_from_size_t(row_major_strides(shape));
}

std::vector<int> runtime::gpu::cudnn_util::compute_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }
    return strides;
}

std::vector<int>
    runtime::gpu::cudnn_util::get_vector_int_from_size_t(const std::vector<size_t>& vec)
{
    std::vector<int> low_vec(vec.size(), 1);
    for (int i = 0; i < vec.size(); i++)
    {
        low_vec[i] = static_cast<int>(vec[i]);
    }
    return low_vec;
}

runtime::gpu::CUDNNEmitter::CUDNNEmitter(GPUPrimitiveEmitter* emitter,
                                         GPURuntimeContext* ctx,
                                         std::shared_ptr<GPUHostParameters> params)
    : m_host_parameters(params)
    , m_primitive_emitter(emitter)
{
    m_ctx = ctx;
}

cudnnDataType_t runtime::gpu::CUDNNEmitter::get_cudnn_datatype(std::string dtype)
{
    static const std::unordered_map<std::string, cudnnDataType_t> datatype_map{
        {"float", CUDNN_DATA_FLOAT},
        {"double", CUDNN_DATA_DOUBLE},
        {"int8_t", CUDNN_DATA_INT8},
        {"int32_t", CUDNN_DATA_INT32}};
    auto p = datatype_map.find(dtype);
    if (p == datatype_map.end())
    {
        std::string err = dtype + "is not supported by cuDNN";
        throw std::runtime_error(err);
    }
    return p->second;
}

size_t runtime::gpu::CUDNNEmitter::build_reduce_forward(const cudnnReduceTensorOp_t& reduce_op,
                                                        const std::string& dtype,
                                                        const Shape& input_shape,
                                                        const AxisSet& reduction_axes)
{
    std::stringstream ss;
    ss << "reduce_op_" << reduce_op << "_dtype_" << dtype << "_i" << join(input_shape, "_") << "_ra"
       << join(reduction_axes, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    auto& desc = m_descriptors.build<cudnnReduceTensorDescriptor_t>();
    cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    auto& input_desc = tensor_descriptor_from_shape(input_shape, data_type, tensor_format);
    Shape output_shape = input_shape;
    // mark reduced axes of input tensor for output tensor descriptor
    for (auto const& idx_dim : reduction_axes)
    {
        output_shape[idx_dim] = 1;
    }
    auto& output_desc = tensor_descriptor_from_shape(output_shape, data_type, tensor_format);

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    size_t workspace_size = 0;
    CUDNN_SAFE_CALL(cudnnGetReductionWorkspaceSize(
        *m_ctx->cudnn_handle, desc, input_desc, output_desc, &workspace_size));
    size_t workspace_idx = allocator.reserve_workspace(workspace_size);
    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);

    // emit reduce operation
    std::unique_ptr<gpu::primitive> reduce(
        new gpu::primitive{[=, &desc, &input_desc, &output_desc](void** inputs, void** outputs) {
            CUDNN_SAFE_CALL(cudnnSetReduceTensorDescriptor(desc,
                                                           reduce_op,
                                                           data_type,
                                                           CUDNN_NOT_PROPAGATE_NAN,
                                                           CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                           CUDNN_32BIT_INDICES));

            void* workspace_ptr = runtime::gpu::invoke_memory_primitive(m_ctx, workspace_idx);
            CUDNN_SAFE_CALL(cudnnReduceTensor(*m_ctx->cudnn_handle,
                                              desc,
                                              nullptr,
                                              0,
                                              workspace_ptr,
                                              workspace_size,
                                              alpha,
                                              input_desc,
                                              inputs[0],
                                              beta,
                                              output_desc,
                                              outputs[0]));
            debug_sync();
        }});

    return this->m_primitive_emitter->register_primitive(reduce, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_tensor_op(const cudnnOpTensorOp_t& tensor_op,
                                                   const std::string& dtype,
                                                   const Shape& input_shape,
                                                   const double alpha0,
                                                   const double alpha1,
                                                   const double beta)
{
    std::stringstream ss;
    ss << "tensor_op" << tensor_op << "_dtype_" << dtype << "_i" << join(input_shape, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    auto& opTensorDesc = m_descriptors.build<cudnnOpTensorDescriptor_t>();
    cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    auto& descriptor = tensor_descriptor_from_shape(input_shape, data_type, tensor_format);

    void* alpha_dt0 = m_host_parameters.allocate_by_datatype(data_type, alpha0);
    void* alpha_dt1 = m_host_parameters.allocate_by_datatype(data_type, alpha1);
    void* beta_dt = m_host_parameters.allocate_by_datatype(data_type, beta);

    // emit tensor binary operation
    std::unique_ptr<gpu::primitive> tensor(
        new gpu::primitive{[=, &opTensorDesc, &descriptor](void** inputs, void** outputs) {
            CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(
                opTensorDesc, tensor_op, data_type, CUDNN_NOT_PROPAGATE_NAN));

            CUDNN_SAFE_CALL(cudnnOpTensor(*m_ctx->cudnn_handle,
                                          opTensorDesc,
                                          alpha_dt0,
                                          descriptor,
                                          inputs[0],
                                          alpha_dt1,
                                          descriptor,
                                          inputs[1],
                                          beta_dt,
                                          descriptor,
                                          outputs[0]));
            debug_sync();
        }});

    return this->m_primitive_emitter->register_primitive(tensor, hash);
}

cudnnFilterDescriptor_t& runtime::gpu::CUDNNEmitter::get_cudnn_filter_descriptor(
    const Shape& shape, const cudnnDataType_t data_type, const cudnnTensorFormat_t tensor_format)
{
    std::vector<int> dimensions(fmax(4, shape.size()), 1);
    int idx = 0;
    for (size_t i = dimensions.size() - shape.size(); i < dimensions.size(); i++)
    {
        dimensions[i] = static_cast<int>(shape[idx++]);
    }

    auto& filter_descriptor = m_descriptors.build<cudnnFilterDescriptor_t>();

    if (dimensions.size() <= 4)
    {
        CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
                                                   /*dataType=*/data_type,
                                                   /*format=*/tensor_format,
                                                   /*dimension_size*/ dimensions[0],
                                                   /*dimension_size*/ dimensions[1],
                                                   /*dimension_size*/ dimensions[2],
                                                   /*dimension_size*/ dimensions[3]));
    }
    else
    {
        CUDNN_SAFE_CALL(
            cudnnSetFilterNdDescriptor(filter_descriptor,
                                       /*dataType=*/data_type,
                                       /*format=*/tensor_format,
                                       /*num_dimensions=*/static_cast<int>(dimensions.size()),
                                       /*dimensions*/ dimensions.data()));
    }
    return filter_descriptor;
}

cudnnConvolutionDescriptor_t& runtime::gpu::CUDNNEmitter::get_cudnn_convolution_descriptor(
    const Shape& padding,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type)
{
    auto& conv_descriptor = m_descriptors.build<cudnnConvolutionDescriptor_t>();
    std::vector<int> window_movement_strides_int(window_movement_strides.size());
    std::vector<int> window_dilation_strides_int(window_dilation_strides.size());
    std::vector<int> padding_int(padding.size());
    for (int i = 0; i < padding.size(); i++)
    {
        window_movement_strides_int[i] = static_cast<int>(window_movement_strides[i]);
        window_dilation_strides_int[i] = static_cast<int>(window_dilation_strides[i]);
        padding_int[i] = static_cast<int>(padding[i]);
    }

    if (padding.size() == 2)
    {
        CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                        padding_int[0],
                                                        padding_int[1],
                                                        window_movement_strides_int[0],
                                                        window_movement_strides_int[1],
                                                        window_dilation_strides_int[0],
                                                        window_dilation_strides_int[1],
                                                        mode,
                                                        data_type));
    }
    else
    {
        CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(conv_descriptor,
                                                        static_cast<int>(padding_int.size()),
                                                        padding_int.data(),
                                                        window_movement_strides_int.data(),
                                                        window_dilation_strides_int.data(),
                                                        mode,
                                                        data_type));
    }
    return conv_descriptor;
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::Convolution* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape = args[0].get_shape();
    auto filter_shape = args[1].get_shape();
    auto output_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides();
    Strides window_movement_strides = node->get_window_movement_strides();
    Strides data_dilation_strides = node->get_data_dilation_strides();
    CoordinateDiff padding_below_diff = node->get_padding_below();
    CoordinateDiff padding_above_diff = node->get_padding_above();
    auto dtype = out[0].get_element_type().c_type_string();

    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "convolution_op_" << dtype << "_i" << join(input_shape, "_") << "_w"
       << join(filter_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
       << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_") << "_p"
       << join(padding_below_diff, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    bool is_deconvolution = false;
    for (auto a : data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }

    bool pad_required = (padding_below_diff != padding_above_diff);

    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }
    Shape input_shape_padded = input_shape;
    Shape padding_interior(data_dilation_strides);

    size_t idx_workspace = std::numeric_limits<size_t>::max();
    size_t pad_dynamic_index = std::numeric_limits<size_t>::max();
    bool can_find_algo = true;
    if (pad_required || is_deconvolution)
    {
        input_shape_padded = runtime::gpu::get_padded_shape(
            input_shape, padding_below, padding_above, padding_interior);
        auto temp_size = shape_size(input_shape_padded) * args[0].get_element_type().size();
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();

        // reserve zero initialized workspace
        idx_workspace = allocator.reserve_workspace(temp_size, true);

        auto& cuda_emitter = m_primitive_emitter->get_cuda_emitter();
        pad_dynamic_index =
            cuda_emitter->build_pad_dynamic({{args[0].get_element_type().c_type_string(),
                                              out[0].get_element_type().c_type_string()}},
                                            input_shape,
                                            input_shape_padded,
                                            padding_below,
                                            padding_interior);

        // asymetric padding has been applied, zero out padding vectors to
        // ensure cudnn does not assume padding
        std::fill(padding_below.begin(), padding_below.end(), 0);
        // padding will make find_algorithm for convolution get wrong result
        can_find_algo = false;
    }

    size_t conv_index = build_convolution(dtype,
                                          input_shape_padded,
                                          filter_shape,
                                          output_shape,
                                          window_movement_strides,
                                          window_dilation_strides,
                                          padding_below,
                                          can_find_algo);

    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            if (idx_workspace != std::numeric_limits<size_t>::max() &&
                pad_dynamic_index != std::numeric_limits<size_t>::max())
            {
                void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                gpu::invoke_primitive(m_ctx,
                                      pad_dynamic_index,
                                      std::vector<void*>{inputs[0]}.data(),
                                      std::vector<void*>{pad_buffer}.data());
                gpu::invoke_primitive(
                    m_ctx, conv_index, std::vector<void*>{pad_buffer, inputs[1]}.data(), outputs);
            }
            else
            {
                gpu::invoke_primitive(m_ctx, conv_index, inputs, outputs);
            }
        }});

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::ConvolutionBackpropData* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape = args[0].get_shape();
    auto filter_shape = args[1].get_shape();
    auto output_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides_forward();
    Strides window_movement_strides = node->get_window_movement_strides_forward();
    Strides data_dilation_strides = node->get_data_dilation_strides_forward();
    CoordinateDiff padding_below_diff = node->get_padding_below_forward();
    CoordinateDiff padding_above_diff = node->get_padding_above_forward();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "convolution_bp_data_op_" << output_type << "_i" << join(input_shape, "_") << "_w"
       << join(filter_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
       << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_") << "_p"
       << join(padding_below_diff, "_");
    std::string hash = ss.str();
    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    bool is_deconvolution = false;
    for (auto a : data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }

    bool pad_required = (padding_below_diff != padding_above_diff);

    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }

    auto output_shape_padded = output_shape;
    Shape padding_below_back(output_shape.size(), 0);
    Shape padding_interior_back(output_shape.size(), 1);
    size_t i = padding_below_back.size() - padding_below.size();
    size_t j = 0;
    for (; i < padding_below_back.size(); i++)
    {
        padding_below_back[i] = padding_below[j];
        padding_interior_back[i] = data_dilation_strides[j];
        j++;
    }

    Shape padding_interior(data_dilation_strides);

    size_t idx_workspace = std::numeric_limits<size_t>::max();
    size_t pad_dynamic_index = std::numeric_limits<size_t>::max();
    size_t slice_index = std::numeric_limits<size_t>::max();
    bool can_find_algo = true;
    if (pad_required || is_deconvolution)
    {
        output_shape_padded =
            get_padded_shape(output_shape, padding_below, padding_above, padding_interior);
        auto temp_size = shape_size(output_shape_padded) * args[0].get_element_type().size();
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();

        // reserve zero initialized workspace
        idx_workspace = allocator.reserve_workspace(temp_size, true);

        auto& cuda_emitter = m_primitive_emitter->get_cuda_emitter();
        pad_dynamic_index = cuda_emitter->build_pad_dynamic({{input_type, output_type}},
                                                            output_shape,
                                                            output_shape_padded,
                                                            padding_below,
                                                            padding_interior);

        slice_index = cuda_emitter->build_slice({{input_type, output_type}},
                                                output_shape_padded,
                                                padding_below_back,
                                                padding_interior_back,
                                                output_shape);

        // asymetric padding has been applied, zero out padding vectors to
        // ensure cudnn does not assume padding
        std::fill(padding_below.begin(), padding_below.end(), 0);
        // padding will make find_algorithm for convolution get wrong result
        can_find_algo = false;
    }

    size_t conv_index = build_convolution_backward_data(output_type,
                                                        args[0].get_shape(),
                                                        args[1].get_shape(),
                                                        output_shape_padded,
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        padding_below,
                                                        can_find_algo);

    std::unique_ptr<gpu::primitive> kernel_launch(new gpu::primitive{[=](void** inputs,
                                                                         void** outputs) mutable {
        if (idx_workspace != std::numeric_limits<size_t>::max() &&
            pad_dynamic_index != std::numeric_limits<size_t>::max() &&
            slice_index != std::numeric_limits<size_t>::max())
        {
            void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
            gpu::invoke_primitive(m_ctx,
                                  pad_dynamic_index,
                                  std::vector<void*>{inputs[0]}.data(),
                                  std::vector<void*>{pad_buffer}.data());
            gpu::invoke_primitive(m_ctx, conv_index, inputs, std::vector<void*>{pad_buffer}.data());
            gpu::invoke_primitive(
                m_ctx, slice_index, std::vector<void*>{pad_buffer}.data(), outputs);
        }
        else
        {
            gpu::invoke_primitive(m_ctx, conv_index, inputs, outputs);
        }
    }});

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::ConvolutionBackpropFilters* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape_0 = args[0].get_shape();
    auto input_shape_1 = args[1].get_shape();
    auto filter_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides_forward();
    Strides window_movement_strides = node->get_window_movement_strides_forward();
    Strides data_dilation_strides = node->get_data_dilation_strides_forward();
    CoordinateDiff padding_below_diff = node->get_padding_below_forward();
    CoordinateDiff padding_above_diff = node->get_padding_above_forward();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list

    std::stringstream ss;
    ss << "convolution_bp_filter_op_" << output_type << "_i" << join(input_shape_0, "_") << "_w"
       << join(filter_shape, "_") << "_o" << join(input_shape_1, "_") << "_ws"
       << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_") << "_p"
       << join(padding_below_diff, "_");
    std::string hash = ss.str();
    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    bool is_deconvolution = false;
    for (auto a : data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }

    bool pad_required = (padding_below_diff != padding_above_diff);

    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }
    auto input_shape_padded = input_shape_0;
    Shape padding_interior(data_dilation_strides);

    size_t idx_workspace = std::numeric_limits<size_t>::max();
    size_t pad_dynamic_index = std::numeric_limits<size_t>::max();
    bool can_find_algo = true;
    if (pad_required || is_deconvolution)
    {
        input_shape_padded = runtime::gpu::get_padded_shape(
            input_shape_0, padding_below, padding_above, padding_interior);
        auto temp_size = shape_size(input_shape_padded) * args[0].get_element_type().size();
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();

        // reserve zero initialized workspace
        idx_workspace = allocator.reserve_workspace(temp_size, true);

        auto& cuda_emitter = m_primitive_emitter->get_cuda_emitter();
        pad_dynamic_index = cuda_emitter->build_pad_dynamic({{input_type, output_type}},
                                                            input_shape_0,
                                                            input_shape_padded,
                                                            padding_below,
                                                            padding_interior);

        // asymetric padding has been applied, zero out padding vectors to
        // ensure cudnn does not assume padding
        std::fill(padding_below.begin(), padding_below.end(), 0);
        // padding will make find_algorithm for convolution get wrong result
        can_find_algo = false;
    }

    size_t conv_index = build_convolution_backward_filter(output_type,
                                                          input_shape_padded,
                                                          input_shape_1,
                                                          filter_shape,
                                                          window_movement_strides,
                                                          window_dilation_strides,
                                                          padding_below,
                                                          can_find_algo);

    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            if (idx_workspace != std::numeric_limits<size_t>::max() &&
                pad_dynamic_index != std::numeric_limits<size_t>::max())
            {
                void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                gpu::invoke_primitive(m_ctx,
                                      pad_dynamic_index,
                                      std::vector<void*>{inputs[0]}.data(),
                                      std::vector<void*>{pad_buffer}.data());
                gpu::invoke_primitive(
                    m_ctx, conv_index, std::vector<void*>{pad_buffer, inputs[1]}.data(), outputs);
            }
            else
            {
                gpu::invoke_primitive(m_ctx, conv_index, inputs, outputs);
            }
        }});

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::MaxPool* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto& input_shape = args[0].get_shape();
    auto& result_shape = out[0].get_shape();
    auto padding_below = node->get_padding_below();
    auto padding_above = node->get_padding_above();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "max_pool_" << output_type << "_i" << join(input_shape, "_") << "_o"
       << join(result_shape, "_") << "_ws" << join(node->get_window_shape(), "_") << "_wst"
       << join(node->get_window_movement_strides(), "_") << "_pb" << join(padding_below, "_")
       << "_pb" << join(padding_above, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    /// assymetric padding detection
    bool pad_required = false;
    auto shape_to_pool =
        runtime::gpu::get_padded_shape(input_shape, padding_below, padding_above, {});
    if (shape_to_pool != input_shape)
    {
        pad_required = true;
    }

    pad_required = pad_required && (padding_below != padding_above);
    // asymetric padding
    size_t idx_workspace = std::numeric_limits<size_t>::max();
    size_t pad_index = std::numeric_limits<size_t>::max();
    if (pad_required)
    {
        auto temp_size = shape_size(shape_to_pool) * args[0].get_element_type().size();
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();
        idx_workspace = allocator.reserve_workspace(temp_size);

        auto pad_value = TypeInfo::Get(args[0].get_element_type())->lowest();

        auto& cuda_emitter = m_primitive_emitter->get_cuda_emitter();
        pad_index = cuda_emitter->build_pad({{input_type, output_type}},
                                            input_shape,
                                            shape_to_pool,
                                            padding_below,
                                            padding_above,
                                            Shape{},
                                            pad_value);

        // asymetric padding has been applied, zero out padding vectors to
        // ensure cuDNN does not assume padding during pooling
        std::fill(padding_below.begin(), padding_below.end(), 0);
        std::fill(padding_above.begin(), padding_above.end(), 0);
    }

    /// end asymmetric padding detection

    size_t max_pool_index = build_pooling(CUDNN_POOLING_MAX,
                                          output_type,
                                          CUDNNEmitter::Prop::Forward,
                                          shape_to_pool,
                                          result_shape,
                                          node->get_window_movement_strides(),
                                          node->get_window_shape(),
                                          padding_below,
                                          padding_above);

    std::unique_ptr<gpu::primitive> kernel_launch(
        new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            if (idx_workspace != std::numeric_limits<size_t>::max() &&
                pad_index != std::numeric_limits<size_t>::max())
            {
                // void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                // gpu::invoke_primitive(m_ctx,
                //                       pad_dynamic_index,
                //                       std::vector<void*>{inputs[0]}.data(),
                //                       std::vector<void*>{pad_buffer}.data());

                // gpu::invoke_primitive(
                //     m_ctx, conv_index, std::vector<void*>{pad_buffer, inputs[1]}.data(), outputs);

                void* pad_buffer = runtime::gpu::invoke_memory_primitive(m_ctx, idx_workspace);
                gpu::invoke_primitive(m_ctx,
                                      pad_index,
                                      std::vector<void*>{inputs[0]}.data(),
                                      std::vector<void*>{pad_buffer}.data());
                gpu::invoke_primitive(
                    m_ctx, max_pool_index, std::vector<void*>{pad_buffer}.data(), outputs);
            }
            else
            {
                gpu::invoke_primitive(m_ctx, max_pool_index, inputs, outputs);
            }
        }});

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::Max* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto& input_shape = args[0].get_shape();
    auto& output_shape = out[0].get_shape();
    auto input_size = shape_size(input_shape);
    auto output_size = shape_size(output_shape);
    auto output_element_size = out[0].get_element_type().size();
    auto output_type = out[0].get_element_type().c_type_string();

    std::stringstream ss;
    ss << "max_" << output_type << "_i" << join(input_shape, "_") << "_ra"
       << join(node->get_reduction_axes(), "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> kernel_launch;
    ;

    // one of args[] axes has zero size, zero output
    if (input_size == 0)
    {
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();
        std::vector<float> negative_inf(output_size, -std::numeric_limits<float>::infinity());
        size_t idx_float_inf =
            allocator.reserve_argspace(negative_inf.data(), negative_inf.size() * sizeof(float));

        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* temp_d = runtime::gpu::invoke_memory_primitive(m_ctx, idx_float_inf);
            runtime::gpu::cuda_memcpyDtD(outputs[0], temp_d, output_size * output_element_size);
        }});
    }
    else if (input_size == output_size)
    {
        // no reduction
        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            runtime::gpu::cuda_memcpyDtD(outputs[0], inputs[0], output_size * output_element_size);
        }});
    }
    else
    {
        auto& cudnn_emitter = m_primitive_emitter->get_cudnn_emitter();
        auto max_index = cudnn_emitter->build_reduce_forward(
            CUDNN_REDUCE_TENSOR_MAX, output_type, input_shape, node->get_reduction_axes());
        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            gpu::invoke_primitive(m_ctx, max_index, inputs, outputs);
        }});
    }

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_primitive(const op::Min* node)
{
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto& input_shape = args[0].get_shape();
    auto& output_shape = out[0].get_shape();
    auto input_size = shape_size(input_shape);
    auto output_size = shape_size(output_shape);
    auto output_element_size = out[0].get_element_type().size();
    auto output_type = out[0].get_element_type().c_type_string();

    std::stringstream ss;
    ss << "min_" << output_type << "_i" << join(input_shape, "_") << "_ra"
       << join(node->get_reduction_axes(), "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    std::unique_ptr<gpu::primitive> kernel_launch;
    ;

    // one of args[] axes has zero size, zero output
    if (input_size == 0)
    {
        GPUAllocator allocator = m_primitive_emitter->get_memory_allocator();
        std::vector<float> negative_inf(output_size, std::numeric_limits<float>::infinity());
        size_t idx_float_inf =
            allocator.reserve_argspace(negative_inf.data(), negative_inf.size() * sizeof(float));

        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            void* temp_d = runtime::gpu::invoke_memory_primitive(m_ctx, idx_float_inf);
            runtime::gpu::cuda_memcpyDtD(outputs[0], temp_d, output_size * output_element_size);
        }});
    }
    else if (input_size == output_size)
    {
        // no reduction
        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            runtime::gpu::cuda_memcpyDtD(outputs[0], inputs[0], output_size * output_element_size);
        }});
    }
    else
    {
        auto& cudnn_emitter = m_primitive_emitter->get_cudnn_emitter();
        auto min_index = cudnn_emitter->build_reduce_forward(
            CUDNN_REDUCE_TENSOR_MIN, output_type, input_shape, node->get_reduction_axes());
        kernel_launch.reset(new gpu::primitive{[=](void** inputs, void** outputs) mutable {
            gpu::invoke_primitive(m_ctx, min_index, inputs, outputs);
        }});
    }

    return this->m_primitive_emitter->register_primitive(kernel_launch, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_convolution(const std::string& dtype,
                                                     const Shape& input_tensor_shape,
                                                     const Shape& input_filter_shape,
                                                     const Shape& output_tensor_shape,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const Shape& padding_below,
                                                     const bool find_algo)
{
    cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto& tensor_desc_0 =
        tensor_descriptor_from_shape(input_tensor_shape, data_type, tensor_format);
    auto& tensor_desc_1 =
        tensor_descriptor_from_shape(output_tensor_shape, data_type, tensor_format);
    auto& filter_desc = get_cudnn_filter_descriptor(input_filter_shape, data_type, tensor_format);
    auto& conv_desc = get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type);

    cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    if (find_algo)
    {
        CUDNN_SAFE_CALL(cudnnGetConvolutionForwardAlgorithm(*m_ctx->cudnn_handle,
                                                            tensor_desc_0,
                                                            filter_desc,
                                                            conv_desc,
                                                            tensor_desc_1,
                                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                            /*memoryLimitInBytes=*/0,
                                                            &conv_fwd_algo));
    }

    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);

    size_t workspace_size_in_bytes = 0;
    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(*m_ctx->cudnn_handle,
                                                            tensor_desc_0,
                                                            filter_desc,
                                                            conv_desc,
                                                            tensor_desc_1,
                                                            conv_fwd_algo,
                                                            &workspace_size_in_bytes));

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    // (lazy) allocation for kernel arguments
    size_t workspace_idx = allocator.reserve_workspace(workspace_size_in_bytes);

    std::unique_ptr<gpu::primitive> conv;
    conv.reset(new gpu::primitive{[=, &conv_desc, &tensor_desc_0, &filter_desc, &tensor_desc_1](
        void** inputs, void** outputs) {
        void* workspace_ptr = runtime::gpu::invoke_memory_primitive(m_ctx, workspace_idx);
        CUDNN_SAFE_CALL(cudnnConvolutionForward(*m_ctx->cudnn_handle,
                                                alpha,
                                                tensor_desc_0,
                                                inputs[0],
                                                filter_desc,
                                                inputs[1],
                                                conv_desc,
                                                conv_fwd_algo,
                                                workspace_ptr,
                                                workspace_size_in_bytes,
                                                beta,
                                                tensor_desc_1,
                                                outputs[0]));
        debug_sync();
    }});

    return this->m_primitive_emitter->insert(std::move(conv));
}

size_t runtime::gpu::CUDNNEmitter::build_convolution_backward_data(
    const std::string& dtype,
    const Shape& input_filter_shape,
    const Shape& input_tensor_shape,
    const Shape& output_tensor_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const Shape& padding_below,
    const bool find_algo)
{
    const cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto& tensor_desc_0 =
        tensor_descriptor_from_shape(input_tensor_shape, data_type, tensor_format);
    auto& tensor_desc_1 =
        tensor_descriptor_from_shape(output_tensor_shape, data_type, tensor_format);
    auto& filter_desc = get_cudnn_filter_descriptor(input_filter_shape, data_type, tensor_format);
    auto& conv_desc = get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type);
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    if (find_algo)
    {
        CUDNN_SAFE_CALL(
            cudnnGetConvolutionBackwardDataAlgorithm(*m_ctx->cudnn_handle,
                                                     filter_desc,
                                                     tensor_desc_0,
                                                     conv_desc,
                                                     tensor_desc_1,
                                                     CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                     0,
                                                     &conv_bwd_data_algo));
    }

    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);

    size_t workspace_size_in_bytes = 0;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(*m_ctx->cudnn_handle,
                                                                 filter_desc,
                                                                 tensor_desc_0,
                                                                 conv_desc,
                                                                 tensor_desc_1,
                                                                 conv_bwd_data_algo,
                                                                 &workspace_size_in_bytes));

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    // (lazy) allocation for kernel arguments
    size_t workspace_idx = allocator.reserve_workspace(workspace_size_in_bytes);

    std::unique_ptr<gpu::primitive> conv;
    conv.reset(new gpu::primitive{[=, &conv_desc, &tensor_desc_0, &filter_desc, &tensor_desc_1](
        void** inputs, void** outputs) {
        void* workspace_ptr = runtime::gpu::invoke_memory_primitive(m_ctx, workspace_idx);
        CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(*m_ctx->cudnn_handle,
                                                     alpha,
                                                     filter_desc,
                                                     inputs[0],
                                                     tensor_desc_0,
                                                     inputs[1],
                                                     conv_desc,
                                                     conv_bwd_data_algo,
                                                     workspace_ptr,
                                                     workspace_size_in_bytes,
                                                     beta,
                                                     tensor_desc_1,
                                                     outputs[0]));
        debug_sync();
    }});

    return this->m_primitive_emitter->insert(std::move(conv));
}

size_t runtime::gpu::CUDNNEmitter::build_convolution_backward_filter(
    const std::string& dtype,
    const Shape& input_tensor_shape_0,
    const Shape& input_tensor_shape_1,
    const Shape& output_filter_shape,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const Shape& padding_below,
    const bool find_algo)
{
    const cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto& tensor_desc_0 =
        tensor_descriptor_from_shape(input_tensor_shape_0, data_type, tensor_format);
    auto& tensor_desc_1 =
        tensor_descriptor_from_shape(input_tensor_shape_1, data_type, tensor_format);
    auto& filter_desc = get_cudnn_filter_descriptor(output_filter_shape, data_type, tensor_format);
    auto& conv_desc = get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type);
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    if (find_algo)
    {
        CUDNN_SAFE_CALL(
            cudnnGetConvolutionBackwardFilterAlgorithm(*m_ctx->cudnn_handle,
                                                       tensor_desc_0,
                                                       tensor_desc_1,
                                                       conv_desc,
                                                       filter_desc,
                                                       CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                       0,
                                                       &conv_bwd_filter_algo));
    }
    size_t workspace_size_in_bytes = 0;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(*m_ctx->cudnn_handle,
                                                                   tensor_desc_0,
                                                                   tensor_desc_1,
                                                                   conv_desc,
                                                                   filter_desc,
                                                                   conv_bwd_filter_algo,
                                                                   &workspace_size_in_bytes));

    // get an allocator for transient per kernel gpu memory
    GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
    // (lazy) allocation for kernel arguments
    size_t workspace_idx = allocator.reserve_workspace(workspace_size_in_bytes);
    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);

    std::unique_ptr<gpu::primitive> conv;
    conv.reset(new gpu::primitive{[=, &conv_desc, &tensor_desc_0, &filter_desc, &tensor_desc_1](
        void** inputs, void** outputs) {
        void* workspace_ptr = runtime::gpu::invoke_memory_primitive(m_ctx, workspace_idx);
        CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(*m_ctx->cudnn_handle,
                                                       alpha,
                                                       tensor_desc_0,
                                                       inputs[0],
                                                       tensor_desc_1,
                                                       inputs[1],
                                                       conv_desc,
                                                       conv_bwd_filter_algo,
                                                       workspace_ptr,
                                                       workspace_size_in_bytes,
                                                       beta,
                                                       filter_desc,
                                                       outputs[0]));
        debug_sync();
    }});

    return this->m_primitive_emitter->insert(std::move(conv));
}

size_t runtime::gpu::CUDNNEmitter::build_pooling(const cudnnPoolingMode_t& pool_op,
                                                 const std::string& dtype,
                                                 const Prop& direction,
                                                 const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 const Strides& window_strides,
                                                 const Shape& window_shape,
                                                 const Shape& padding_below,
                                                 const Shape& padding_above)
{
    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "pool_op" << pool_op << "dtype_" << dtype << "_dir" << static_cast<int>(direction) << "_i"
       << join(input_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
       << join(window_shape, "_") << "_wst" << join(window_strides, "_") << "_pb"
       << join(padding_below, "_") << "_pb" << join(padding_above, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    const cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    auto& desc = m_descriptors.build<cudnnPoolingDescriptor_t>();
    auto& input_desc = tensor_descriptor_from_shape(input_shape, data_type, tensor_format);
    auto& output_desc = tensor_descriptor_from_shape(output_shape, data_type, tensor_format);

    if (input_shape.size() == 3)
    {
    }
    else if (input_shape.size() == 4)
    {
        CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,
                                                    pool_op,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    static_cast<int>(window_shape[0]),
                                                    static_cast<int>(window_shape[1]),
                                                    static_cast<int>(padding_below[0]),
                                                    static_cast<int>(padding_below[1]),
                                                    static_cast<int>(window_strides[0]),
                                                    static_cast<int>(window_strides[1])));
    }
    else if (input_shape.size() == 5)
    {
        std::vector<int> w_strides(window_strides.size());
        std::vector<int> w_shape(window_shape.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_strides[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }
        CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc,
                                                    pool_op,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    3,
                                                    w_shape.data(),
                                                    w_padding.data(),
                                                    w_strides.data()));
    }
    else
    {
        throw std::runtime_error("Pooling currently supports up to 3 spatial dimensions only.");
    }

    std::unique_ptr<gpu::primitive> pool;
    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);

    switch (direction)
    {
    case (Prop::Inference):
    case (Prop::Forward):
    {
        pool.reset(new gpu::primitive{
            [=, &desc, &input_desc, &output_desc](void** inputs, void** outputs) {
                CUDNN_SAFE_CALL(cudnnPoolingForward(*m_ctx->cudnn_handle,
                                                    desc,
                                                    alpha,
                                                    input_desc,
                                                    inputs[0],
                                                    beta,
                                                    output_desc,
                                                    outputs[0]));
                debug_sync();
            }});
        break;
    }
    case (Prop::Backward):
    {
        if (data_type == CUDNN_DATA_INT8 || data_type == CUDNN_DATA_INT32)
        {
            throw std::runtime_error("Pooling does not support int type by cuDNN.");
        }
        pool.reset(new gpu::primitive{
            [=, &desc, &input_desc, &output_desc](void** inputs, void** outputs) {
                // cuDNN requires the output tensor of the maxpool fprop to be passed even though
                // it is not mathematically necessary. It appears, however, that it is not actually
                // used as the adjoints are passed in place and the correct result is achieved.
                CUDNN_SAFE_CALL(cudnnPoolingBackward(*m_ctx->cudnn_handle,
                                                     desc,
                                                     alpha,
                                                     // output (wrt maxpool) tensor
                                                     output_desc,
                                                     inputs[1],
                                                     // adjoint of output
                                                     output_desc,
                                                     inputs[1],
                                                     // input (wrt maxpool) tensor
                                                     input_desc,
                                                     inputs[0],
                                                     beta,
                                                     // adjoint of input
                                                     input_desc,
                                                     outputs[0]));
                debug_sync();
            }});
        break;
    }
    }

    return this->m_primitive_emitter->register_primitive(pool, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_batchnorm(const cudnnBatchNormMode_t& bn_op,
                                                   const std::string& dtype,
                                                   const Prop& direction,
                                                   const Shape& tensor_shape,
                                                   const Shape& param_shape,
                                                   double epsilon)
{
    // Assumes NC{d1...dN} format
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::digits10 + 2);

    ss << "bn_op" << bn_op << "_dtype_" << dtype << "_dir" << static_cast<int>(direction) << "_ts"
       << join(tensor_shape, "_") << "_ps" << join(param_shape, "_") << "_eps" << epsilon;
    std::string hash = ss.str();
    std::replace(hash.begin(), hash.end(), '.', '_');

    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    if (epsilon < CUDNN_BN_MIN_EPSILON)
    {
        throw std::runtime_error("Batch Norm epsilon is less than CUDNN_BN_MIN_EPSILON");
    }

    const cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    auto& derived_param_desc = m_descriptors.build<cudnnTensorDescriptor_t>();
    auto& tensor_desc = tensor_descriptor_from_shape(tensor_shape, data_type, tensor_format);
    CUDNN_SAFE_CALL(cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, bn_op));
    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);
    std::unique_ptr<gpu::primitive> batchnorm;
    switch (direction)
    {
    case Prop::Inference:
    {
        batchnorm.reset(new gpu::primitive{
            [=, &tensor_desc, &derived_param_desc](void** inputs, void** outputs) {
                CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardInference(*m_ctx->cudnn_handle,
                                                                        bn_op,
                                                                        alpha,
                                                                        beta,
                                                                        tensor_desc,
                                                                        inputs[2], // tensor
                                                                        tensor_desc,
                                                                        outputs[0], // tensor
                                                                        derived_param_desc,
                                                                        inputs[0], // gain
                                                                        inputs[1], // bias
                                                                        inputs[3], // mean
                                                                        inputs[4], // variance
                                                                        epsilon));
                debug_sync();
            }});
        break;
    }
    case Prop::Forward:
    {
        auto& op_desc = m_descriptors.build<cudnnOpTensorDescriptor_t>();
        CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(
            op_desc, CUDNN_OP_TENSOR_MUL, data_type, CUDNN_NOT_PROPAGATE_NAN));

        // currently not using the cuDNN moving average
        // calculation so this factor needs to be set to 1.0
        double exp_avg_factor = 1.0;
        // factor to convert unbiased variance to biased variance estimate
        // mini-batch statistics (variance of the sample) should be used
        // in training and population statistics (sample variance) used
        // during inference. see commit note for 3b081ce for more details.
        double m = shape_size(tensor_shape) / tensor_shape[1];
        void* bias_factor = m_host_parameters.allocate_by_datatype(data_type, (m - 1) / m);
        batchnorm.reset(new gpu::primitive{
            [=, &op_desc, &tensor_desc, &derived_param_desc](void** inputs, void** outputs) {
                CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardTraining(*m_ctx->cudnn_handle,
                                                                       bn_op,
                                                                       alpha,
                                                                       beta,
                                                                       tensor_desc,
                                                                       inputs[2],
                                                                       tensor_desc,
                                                                       outputs[0],
                                                                       derived_param_desc,
                                                                       inputs[0],
                                                                       inputs[1],
                                                                       exp_avg_factor,
                                                                       outputs[1],
                                                                       outputs[2],
                                                                       epsilon,
                                                                       NULL,
                                                                       NULL));
                debug_sync();

                // convert to biased variance
                CUDNN_SAFE_CALL(cudnnOpTensor(*m_ctx->cudnn_handle,
                                              op_desc,
                                              beta,
                                              derived_param_desc,
                                              outputs[2],
                                              beta,
                                              derived_param_desc,
                                              outputs[2],
                                              bias_factor,
                                              derived_param_desc,
                                              outputs[2]));
                debug_sync();
            }});
        break;
    }
    case Prop::Backward:
    {
        batchnorm.reset(new gpu::primitive{
            [=, &tensor_desc, &derived_param_desc](void** inputs, void** outputs) {
                CUDNN_SAFE_CALL(cudnnBatchNormalizationBackward(
                    *m_ctx->cudnn_handle,
                    bn_op,
                    alpha,
                    beta,
                    alpha,
                    beta,
                    tensor_desc,
                    inputs[2 /* input tensor x */],
                    tensor_desc,
                    inputs[5 /* dy */],
                    tensor_desc,
                    outputs[0 /* dx */],
                    derived_param_desc,
                    inputs[0 /* gamma */],
                    outputs[1 /* dgamma */],
                    outputs[2 /* dbeta */],
                    epsilon,
                    NULL,   // inputs[3 /* mu batch mean*/],
                    NULL)); // inputs[4 /* 1/sig**2 batch inverse variance*/]);
                debug_sync();
            }});
        break;
    }
    }

    return this->m_primitive_emitter->register_primitive(batchnorm, hash);
}

size_t runtime::gpu::CUDNNEmitter::build_softmax(const cudnnSoftmaxAlgorithm_t& algorithm,
                                                 const cudnnSoftmaxMode_t& mode,
                                                 const std::string& dtype,
                                                 const Prop& direction,
                                                 const Shape& tensor_shape)
{
    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "softmax_op_" << mode << "_dtype_" << dtype << "_alg" << algorithm << "_dir"
       << static_cast<int>(direction) << "_s" << join(tensor_shape, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    cudnnDataType_t data_type = get_cudnn_datatype(dtype);
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    auto& tensor_desc = tensor_descriptor_from_shape(tensor_shape, data_type, tensor_format);
    void* alpha = m_host_parameters.allocate_by_datatype(data_type, 1.0);
    void* beta = m_host_parameters.allocate_by_datatype(data_type, 0);
    std::unique_ptr<runtime::gpu::primitive> softmax;
    switch (direction)
    {
    case Prop::Forward:
    case Prop::Inference:
    {
        softmax.reset(new gpu::primitive{[=, &tensor_desc](void** inputs, void** outputs) {
            CUDNN_SAFE_CALL(cudnnSoftmaxForward(*m_ctx->cudnn_handle,
                                                algorithm,
                                                mode,
                                                alpha,
                                                tensor_desc,
                                                inputs[0],
                                                beta,
                                                tensor_desc,
                                                outputs[0]));
            debug_sync();
        }});
        break;
    }
    case Prop::Backward:
    {
        softmax.reset(new gpu::primitive{[=, &tensor_desc](void** inputs, void** outputs) {
            CUDNN_SAFE_CALL(cudnnSoftmaxBackward(*m_ctx->cudnn_handle,
                                                 algorithm,
                                                 mode,
                                                 alpha,
                                                 tensor_desc,
                                                 inputs[0],
                                                 tensor_desc,
                                                 inputs[1],
                                                 beta,
                                                 tensor_desc,
                                                 outputs[0]));
            debug_sync();
        }});
        break;
    }
    }

    return this->m_primitive_emitter->register_primitive(softmax, hash);
}

void runtime::gpu::CUDNNEmitter::sync()
{
    CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
    return;
}

void runtime::gpu::CUDNNEmitter::debug_sync()
{
#ifdef NGRAPH_DEBUG_ENABLE
    CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return;
}
