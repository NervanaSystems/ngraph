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

#include <sstream>

#include "ngraph/op/reverse.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Reverse reverses the selected axes within a tensor.
            template <>
            void Impl<op::Reverse>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                const auto& shape = op().get_shape();

                set_output(start_tile_function()
                               .add(builder::Input{op_input(), "I"}.add_dims("D", 0, shape.size()))
                               .add(builder::Output{"O"})
                               .add(builder::UnaryContraction{"="}
                                        .set(builder::ContractionOutput{"O"}
                                                 .add_indices("d", 0, shape.size())
                                                 .add_dims("D", 0, shape.size()))
                                        .set(builder::ContractionInput{"I"}.add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < shape.size(); ++idx)
                                            {
                                                auto sidx = std::to_string(idx);
                                                if (op().get_reversed_axes().count(idx))
                                                {
                                                    out = "D" + sidx + "-d" + sidx + "-1";
                                                }
                                                else
                                                {
                                                    out = "d" + sidx;
                                                }
                                            }
                                        })))
                               .finalize());
            }

            namespace
            {
                Impl<op::Reverse>::Registration register_reverse;
            }
        }
    }
}
