/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#pragma once

#include <functional>
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPURuntimeContext;

            namespace cudnn_util
            {
                std::vector<int> compute_strides(const std::vector<int>& dim);
            }

            class CUDNNEmitter
            {
            public:
                CUDNNEmitter() {}
                ~CUDNNEmitter() {}

                size_t build_reduce_forward(GPURuntimeContext* ctx,
                                            const Shape& input_shape,
                                            const AxisSet& reduction_axes,
                                            const cudnnReduceTensorOp_t& reduce_op);

                void invoke(size_t primitive_index, const std::vector<void*>& args, const std::vector<void*>& result);

            private:
                size_t register_primitive(const std::function<void(std::vector<void*>,std::vector<void*>)>& f);
                std::vector<std::function<void(std::vector<void*>,std::vector<void*>)>> m_cudnn_primitives;
            };
        }

    }
}
