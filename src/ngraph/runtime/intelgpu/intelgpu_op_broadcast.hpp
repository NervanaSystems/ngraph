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
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            // This implements Broadcast and Sum nGraph operations.
            // input_shape (bcast) or output_shape (sum) can be empty.
            // If the shape is empty it means scalar
            void do_bcast_sum_operation(cldnn::topology& topology,
                                        const std::string& input_name,
                                        const Shape& input_shape,
                                        const element::Type& input_type,
                                        const std::string& output_name,
                                        const Shape& output_shape,
                                        const element::Type& output_type,
                                        const AxisSet& axis,
                                        bool is_bcast);

            // This implements Min and Max operations depends on is_min parameter
            void do_max_min_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& axis,
                                      bool is_min);

            // This implements Product operation
            void do_product_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& axis);
        }
    }
}
