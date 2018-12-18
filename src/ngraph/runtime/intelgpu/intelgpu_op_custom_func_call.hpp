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

#pragma once

#include <CPP/topology.hpp>

#include "ngraph/axis_set.hpp"
#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            void do_all_any_op(cldnn::topology& topology,
                               const std::string& input0_name,
                               const Shape& input0_shape,
                               const std::string& output_name,
                               const Shape& output_shape,
                               const element::Type& output_type,
                               const AxisSet& axis,
                               const std::string& operation,
                               const std::string& init_val);

            void do_reduce_func_call(cldnn::topology& topology,
                                     const std::string& input0_name,
                                     const Shape& input0_shape,
                                     const std::string& input1_name,
                                     const Shape& input1_shape,
                                     const std::string& output_name,
                                     const Shape& output_shape,
                                     const element::Type& output_type,
                                     const AxisSet& axis,
                                     std::vector<std::shared_ptr<Function>>& func);
        }
    }
}
