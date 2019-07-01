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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            std::shared_ptr<Node>
                QuantizedLinearMatmul(const std::shared_ptr<Node>& input0,
                                      const std::shared_ptr<Node>& input1,
                                      const std::shared_ptr<Node>& input0_scale,
                                      const std::shared_ptr<Node>& input0_zero_point,
                                      const std::shared_ptr<Node>& input1_scale,
                                      const std::shared_ptr<Node>& input1_zero_point,
                                      const std::shared_ptr<Node>& output_scale,
                                      const std::shared_ptr<Node>& output_zero_point);

            std::shared_ptr<Node> QuantizedLinearMatmulInteger(const std::shared_ptr<Node>& input0,
                                                               const std::shared_ptr<Node>& input1);

            std::shared_ptr<Node>
                QuantizedLinearMatmulInteger(const std::shared_ptr<Node>& input0,
                                             const std::shared_ptr<Node>& input1,
                                             const std::shared_ptr<Node>& input0_zero_point,
                                             const std::shared_ptr<Node>& input1_zero_point);
        }
    }
}
