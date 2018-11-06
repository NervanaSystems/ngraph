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

#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Softmax implements a standard ML softmax operation.
            template <>
            void Impl<op::Softmax>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                const auto& shape = op().get_inputs()[0].get_shape();
                auto dim_limit = shape.size();

                auto f = start_tile_function();
                f.add(builder::Input{op_input(0), "I"}.add_dims("D", 0, dim_limit))
                    .add(builder::Output{"O"});

                bool reorder_needed = false;
                bool saw_element = false;
                auto groups = 1;
                auto elements = 1;
                std::vector<std::size_t> group_idxs;
                std::vector<std::size_t> element_idxs;

                for (auto didx = 0; didx < shape.size(); ++didx)
                {
                    if (op().get_axes().count(didx))
                    {
                        elements *= shape[didx];
                        element_idxs.push_back(didx);
                        saw_element = true;
                    }
                    else
                    {
                        groups *= shape[didx];
                        group_idxs.push_back(didx);
                        if (saw_element)
                        {
                            reorder_needed = true;
                        }
                    }
                }

                const char* input = "I";
                const char* output = "O";
                const char* reshape_output = output;
                bool reshape_needed = dim_limit != 2;

                if (!reorder_needed)
                {
                    reshape_needed |= shape[0] != groups;
                }
                else
                {
                    f.add(builder::UnaryContraction{"="}
                              .set(builder::ContractionOutput{"RI"}
                                       .add_dims([&](
                                           std::back_insert_iterator<std::list<std::string>> out) {
                                           for (auto idx : group_idxs)
                                           {
                                               out = "D" + std::to_string(idx);
                                           }
                                           for (auto idx : element_idxs)
                                           {
                                               out = "D" + std::to_string(idx);
                                           }
                                       })
                                       .add_indices([&](
                                           std::back_insert_iterator<std::list<std::string>> out) {
                                           for (auto idx : group_idxs)
                                           {
                                               out = "d" + std::to_string(idx);
                                           }
                                           for (auto idx : element_idxs)
                                           {
                                               out = "d" + std::to_string(idx);
                                           }
                                       }))
                              .set(builder::ContractionInput{"I"}.add_indices("d", 0, dim_limit)));
                    input = "RI";
                    output = "RO";
                    if (group_idxs.size())
                    {
                        reshape_needed |= shape[group_idxs[0]] != groups;
                    }
                    else
                    {
                        reshape_needed |= shape[element_idxs[0]] != groups;
                    }
                }

                if (reshape_needed)
                {
                    std::ostringstream reshape;
                    reshape << "reshape(" << input << ", " << groups << ", " << elements << ")";
                    f.add(builder::Elementwise{"GI", reshape.str()});
                    input = "GI";
                    reshape_output = output;
                    output = "GO";
                }

                {
                    // Take the softmax.
                    std::ostringstream softmax;
                    softmax << "builtin_softmax(" << input << ", " << groups << ", " << elements
                            << ")";
                    f.add(builder::Elementwise{output, softmax.str()});
                }

                if (reshape_needed)
                {
                    // Unbundle the axes.
                    std::ostringstream reshape;
                    reshape << "reshape(GO";
                    for (auto didx : group_idxs)
                    {
                        reshape << ", " << shape[didx];
                    }
                    for (auto didx : element_idxs)
                    {
                        reshape << ", " << shape[didx];
                    }
                    reshape << ")";
                    f.add(builder::Elementwise{reshape_output, reshape.str()});
                    output = reshape_output;
                }

                if (reorder_needed)
                {
                    f.add(builder::UnaryContraction{"="}
                              .set(builder::ContractionOutput{"O"}
                                       .add_dims("D", 0, dim_limit)
                                       .add_indices("d", 0, dim_limit))
                              .set(builder::ContractionInput{output}.add_indices(
                                  [&](std::back_insert_iterator<std::list<std::string>> out) {
                                      for (auto idx : group_idxs)
                                      {
                                          out = "d" + std::to_string(idx);
                                      }
                                      for (auto idx : element_idxs)
                                      {
                                          out = "d" + std::to_string(idx);
                                      }
                                  })));
                }

                set_output(f.finalize());
            }

            namespace
            {
                Impl<op::Softmax>::Registration register_softmax;
            }
        }
    }
}
