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

#pragma once

#include <CPP/topology.hpp>

#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            // This implements BatchNorm nGraph operation
            // nGraph uses channels in this operation but clDNN uses full input data
            void do_batch_norm_operation(cldnn::topology& topology,
                                         const std::string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type,
                                         double eps,
                                         const std::string& input_name,
                                         const Shape& input_shape,
                                         const std::string& gamma_name,
                                         const Shape& gamma_shape,
                                         const std::string& beta_name,
                                         const std::string& mean_name,
                                         const std::string& variance_name);

            // This creates mean of the input matrix by Channel axis
            void do_create_mean(cldnn::topology& topology,
                                const std::string& output_name,
                                const Shape& output_shape,
                                const element::Type& output_type,
                                const std::string& input_name,
                                const Shape& input_shape);

            // This creates mean of the input matrix by Channel axis
            void do_create_variance(cldnn::topology& topology,
                                    const std::string& output_name,
                                    const Shape& output_shape,
                                    const element::Type& output_type,
                                    const std::string& input_name,
                                    const Shape& input_shape,
                                    const std::string& mean_name);
        }
    }
}
