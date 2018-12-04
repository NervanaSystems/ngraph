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

#include "ngraph/log.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Dot is a generalized dot product operation -- scalar-tensor,
            // matrix-vector, and matrix multiplication.
            template <>
            void Impl<op::Dot>::operator()()
            {
                check_inputs(2);
                check_outputs(1);

                auto l_dim_limit = op().get_inputs()[0].get_shape().size();
                auto r_dim_limit = op().get_inputs()[1].get_shape().size();
                auto reduce_limit = op().get_reduction_axes_count();
                auto l_dim_mac = l_dim_limit - reduce_limit;
                auto r_dim_mic = reduce_limit;

                NGRAPH_DEBUG << "l_dim_limit=" << l_dim_limit;
                NGRAPH_DEBUG << "r_dim_limit=" << r_dim_limit;
                NGRAPH_DEBUG << "reduce_limit=" << reduce_limit;
                NGRAPH_DEBUG << "l_dim_mac=" << l_dim_mac;
                NGRAPH_DEBUG << "r_dim_mic=" << r_dim_mic;

                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(0), "L"}
                                 .add_dims("DL", 1, l_dim_mac + 1)
                                 .add_dims("DC", 1, reduce_limit + 1))
                        .add(builder::Input{op_input(1), "R"}
                                 .add_dims("DC", 1, reduce_limit + 1)
                                 .add_dims("DR", r_dim_mic + 1, r_dim_limit + 1))
                        .add(builder::Output{"O"})
                        .add(builder::BinaryContraction{"+", "*"}
                                 .set(builder::ContractionOutput{"O"}
                                          .add_indices("dl", 1, l_dim_mac + 1)
                                          .add_indices("dr", r_dim_mic + 1, r_dim_limit + 1)
                                          .add_dims("DL", 1, l_dim_mac + 1)
                                          .add_dims("DR", r_dim_mic + 1, r_dim_limit + 1))
                                 .set_lhs(builder::ContractionInput{"L"}
                                              .add_indices("dl", 1, l_dim_mac + 1)
                                              .add_indices("dc", 1, reduce_limit + 1))
                                 .set_rhs(builder::ContractionInput{"R"}
                                              .add_indices("dc", 1, reduce_limit + 1)
                                              .add_indices("dr", r_dim_mic + 1, r_dim_limit + 1)))
                        .finalize());
            }

            namespace
            {
                Impl<op::Dot>::Registration register_dot;
            }
        }
    }
}
