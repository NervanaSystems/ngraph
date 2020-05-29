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

#include "matmul_integer.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/log.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector matmul_integer(const Node& node)
                {
                    auto ng_inputs = node.get_ng_inputs();
                    auto factory = builder::MatmulIntegerFactory(
                        OutputVector(std::begin(ng_inputs), std::end(ng_inputs)));
                    std::size_t left_rank{ng_inputs.at(0)->get_shape().size()};
                    std::size_t right_rank{ng_inputs.at(1)->get_shape().size()};

                    return as_node_vector(factory.make_matmul_op());
                }
            }
        }
    }
}
