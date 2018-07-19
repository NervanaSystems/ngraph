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

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            // This implements BatchNorm nGraph operation
            // Since nGraph uses channels in this operation but clDNN uses full input data
            // at one time we have to use following algorithm:
            // 1. Split all input data arrays into several matrices by channel axis
            // 2. Independently do cldnn::batch_norm on particular matrix
            // 3. Every result of the cldnn::batch_norm must be scaled and
            //    shifted because cldnn::batch_norm dosn't use gamma and beta
            // 4. Concatenate all results into output matrix by channel axis
            void do_batch_norm_operation(cldnn::topology& topology,
                                         const std::string& output_name,
                                         double eps,
                                         const std::string& input_name,
                                         const Shape& input_shape,
                                         const std::string& gamma_name,
                                         const std::string& beta_name,
                                         const std::string& mean_name = std::string(),
                                         const std::string& variance_name = std::string());
        }
    }
}
