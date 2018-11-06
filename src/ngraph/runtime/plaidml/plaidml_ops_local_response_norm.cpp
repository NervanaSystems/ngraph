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

#include "ngraph/op/lrn.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // LRN implements Local Response Normalization
            template <>
            void Impl<op::LRN>::operator()()
            {
                check_inputs(1);
                check_outputs(1);
                auto dim_limit = op().get_inputs()[0].get_shape().size();
                auto rank = dim_limit - 2;
                auto distance = op().get_nsize() / 2;
                std::ostringstream div_expr;
                div_expr << "I / pow(" << op().get_bias() << ".0 + ((" << op().get_alpha()
                         << ".0 / " << op().get_nsize() << ".0) * S), " << op().get_beta() << ".0)";
                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(), "I"}
                                 .add_dims({"N", "C"})
                                 .add_dims("D", 0, rank))
                        .add(builder::Output{"O"})
                        .add(builder::Elementwise{"ISQ", "I * I"})
                        .add(builder::UnaryContraction{"+"}
                                 .set(builder::ContractionOutput{"S"}
                                          .add_indices({"n", "c"})
                                          .add_indices("d", 0, rank)
                                          .add_dims({"N", "C"})
                                          .add_dims("D", 0, rank))
                                 .set(builder::ContractionInput{"ISQ"}
                                          .add_indices({"n", "c + z - " + std::to_string(distance)})
                                          .add_indices("d", 0, rank))
                                 .add_constraints(
                                     [&](std::back_insert_iterator<std::list<std::string>> out) {
                                         out = "z < " + std::to_string(op().get_nsize());
                                     }))
                        .add(builder::Elementwise{"O", div_expr.str()})
                        .finalize());
            }

            namespace
            {
                Impl<op::LRN>::Registration register_local_response_norm;
            }
        }
    }
}
