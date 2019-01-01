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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace kernel
            {
                void emit_memset(codegen::CodeWriter& writer,
                                 const GPUTensorWrapper& dst,
                                 int value,
                                 size_t buffer_size = 0);

                void emit_memcpyDtD(codegen::CodeWriter& writer,
                                    const GPUTensorWrapper& dst,
                                    const GPUTensorWrapper& src,
                                    size_t buffer_size = 0);

                void emit_cudnnConvolutionDescriptor(codegen::CodeWriter& writer,
                                                     const std::string& name,
                                                     const CoordinateDiff& padding,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const std::string& mode,
                                                     const std::string& data_type);

                void emit_cudnnFilterDescriptor(codegen::CodeWriter& writer,
                                                const std::string& name,
                                                const std::string& format,
                                                const std::string& data_type,
                                                const Shape& shape);

                void emit_cudnnTensorDescriptor(codegen::CodeWriter& writer,
                                                const std::string& name,
                                                const std::string& format,
                                                const std::string& data_type,
                                                const Shape& shape);

                void emit_cudnnTensor4dDescriptor(codegen::CodeWriter& writer,
                                                  const std::string& name,
                                                  const std::string& format,
                                                  const std::string& data_type,
                                                  const std::array<size_t, 4>& axes);

                void emit_cudnnTensorNdDescriptor(codegen::CodeWriter& writer,
                                                  const std::string& name,
                                                  const std::string& data_type,
                                                  const size_t& num_axes,
                                                  const std::vector<size_t>& axes,
                                                  const std::vector<size_t>& strides);

                void emit_cudnnReduceTensor(codegen::CodeWriter& writer,
                                            const GPUTensorWrapper& in,
                                            const GPUTensorWrapper& out,
                                            const std::string& reduce_op,
                                            const std::string& data_type,
                                            const std::string& nan_prop,
                                            const std::string& input_desc,
                                            const std::string& output_desc,
                                            const float& alpha,
                                            const float& beta);

                std::string emit_type_string(const Node* node);
            }
        }
    }
}
