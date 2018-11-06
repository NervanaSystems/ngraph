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

#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Concat views a tensor as a new type.
            template <>
            void Impl<op::Concat>::operator()()
            {
                check_outputs(1);

                auto f = start_tile_function();
                f.add(builder::Output{"O"});
                std::size_t dim_count = op().get_shape().size();
                std::ostringstream offset;
                std::ostringstream oexpr;
                std::ostringstream concat_dsize;
                bool saw_non_zero_tensor = false;
                for (std::size_t iidx = 0; iidx < op().get_inputs().size(); ++iidx)
                {
                    if (!shape_size(op().get_input_shape(iidx)))
                    {
                        continue;
                    }
                    if (saw_non_zero_tensor)
                    {
                        concat_dsize << "+";
                    }
                    saw_non_zero_tensor = true;
                    concat_dsize << "I" << iidx << "_D" << op().get_concatenation_axis();
                }

                saw_non_zero_tensor = false;
                for (std::size_t iidx = 0; iidx < op().get_inputs().size(); ++iidx)
                {
                    if (!shape_size(op().get_input_shape(iidx)))
                    {
                        continue;
                    }
                    std::string sidx{std::to_string(iidx)};
                    f.add(builder::Input{op_input(iidx), "I" + sidx}.add_dims(
                        "I" + sidx + "_D", 0, dim_count));
                    f.add(builder::UnaryContraction{"="}
                              .set(builder::ContractionOutput{"E" + sidx}
                                       .add_dims([&](
                                           std::back_insert_iterator<std::list<std::string>> out) {
                                           for (std::size_t idx = 0; idx < dim_count; ++idx)
                                           {
                                               std::ostringstream s;
                                               if (idx == op().get_concatenation_axis())
                                               {
                                                   out = concat_dsize.str();
                                               }
                                               else
                                               {
                                                   s << "I" << iidx << "_D" << idx;
                                                   out = s.str();
                                               }
                                           }
                                       })
                                       .add_indices([&](
                                           std::back_insert_iterator<std::list<std::string>> out) {
                                           for (std::size_t idx = 0; idx < dim_count; ++idx)
                                           {
                                               std::ostringstream s;
                                               s << "d" << idx;
                                               if (saw_non_zero_tensor &&
                                                   idx == op().get_concatenation_axis())
                                               {
                                                   s << " + " << offset.str();
                                               }
                                               out = s.str();
                                           }
                                       }))
                              .set(builder::ContractionInput{"I" + sidx}.add_indices(
                                  "d", 0, dim_count)));
                    if (saw_non_zero_tensor)
                    {
                        oexpr << " + ";
                        offset << " + ";
                    }
                    oexpr << "E" << sidx;
                    offset << "I" << iidx << "_D" << op().get_concatenation_axis();
                    saw_non_zero_tensor = true;
                }
                f.add(builder::Elementwise{"O", oexpr.str()});

                set_output(f.finalize());
            }

            namespace
            {
                Impl<op::Concat>::Registration register_concat;
            }
        }
    }
}
