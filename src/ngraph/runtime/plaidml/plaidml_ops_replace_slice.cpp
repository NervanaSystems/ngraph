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

#include "ngraph/op/replace_slice.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // ReplaceSlice replaces part of a tensor with another tensor.
            template <>
            void Impl<op::ReplaceSlice>::operator()()
            {
                check_inputs(2);
                check_outputs(1);

                // For ReplaceSlice:
                //
                // * Pad the second tensor to match the first (same-size dimensions and offset according to the
                // * lower bounds of the replacement, with the desired stridings)
                //
                // * Generate a boolean tensor of the same shape as the first, where true == "Do the
                // * replacement".
                //
                // * Use a trinary to do the replacement.

                const auto& shape = op().get_shape();

                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(0), "L"}.add_dims("D", 0, shape.size()))
                        .add(builder::Input{op_input(1), "S"}.add_dims("SD", 0, shape.size()))
                        .add(builder::Output{"O"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"O"}
                                        .add_dims("D", 0, shape.size())
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < shape.size(); ++idx)
                                            {
                                                auto stride = op().get_strides()[idx];
                                                auto lower_bound = op().get_lower_bounds()[idx];
                                                std::ostringstream didx;
                                                if ((stride != 1) && lower_bound)
                                                {
                                                    didx << "(";
                                                }
                                                didx << "d" << idx;
                                                if (stride != 1)
                                                {
                                                    didx << "*" << stride;
                                                }
                                                if ((stride != 1) && lower_bound)
                                                {
                                                    didx << ")";
                                                }
                                                if (lower_bound)
                                                {
                                                    didx << "+" << lower_bound;
                                                }
                                                out = didx.str();
                                            }
                                        }))
                                .set(builder::ContractionInput{"S"}.add_indices(
                                    "d", 0, shape.size()))
                                .add_constraints(
                                    [&](std::back_insert_iterator<std::list<std::string>> out) {
                                        for (std::size_t idx = 0; idx < shape.size(); ++idx)
                                        {
                                            out = "d" + std::to_string(idx) + " < " +
                                                  std::to_string(op().get_upper_bounds()[idx] -
                                                                 op().get_lower_bounds()[idx]);
                                        }
                                    })
                                .set_default("L"))
                        .finalize());
            }

            namespace
            {
                Impl<op::ReplaceSlice>::Registration register_replace_slice;
            }
        }
    }
}
