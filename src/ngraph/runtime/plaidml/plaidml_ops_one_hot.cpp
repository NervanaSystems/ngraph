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

#include "ngraph/op/one_hot.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // OneHot performs one-hot encoding along the requested axis.
            template <>
            void Impl<op::OneHot>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                // Here's what's going on to implement OneHot:
                //
                // * We reshape the input tensor to add a size=1 dimension where we want the one-hot axis to be,
                //
                // * We create an index tensor that's size=1 on every dimension except the one-hot dimension,
                //
                // * We perform an elementwise conditional across them to assign the one-hot values.
                //
                // The broadcast rules will expand the index tensor on all non-one-hot dimensions to match the
                // input, and will expand the input tensor on the one-hot dimension to match the index.
                //
                // In theory, it'd be pretty easy to implement all this with purely elementwise operations.  The
                // current definition of index() requires an input tensor of the index() output shape, and it's
                // a little tricky to fix that, so we generate a zero tensor of the correct shape using a
                // contraction.  TODO: Optimize out the zero tensor contraction.

                const auto& in_shape = op().get_inputs()[0].get_shape();
                const auto& out_shape = op().get_shape();

                std::ostringstream in_reshape;
                for (std::size_t idx = 0; idx < out_shape.size(); ++idx)
                {
                    if (idx)
                    {
                        in_reshape << ", ";
                    }
                    if (idx == op().get_one_hot_axis())
                    {
                        in_reshape << 1;
                    }
                    else
                    {
                        in_reshape << out_shape[idx];
                    }
                }

                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(), "I"}.add_dims("D", 0, in_shape.size()))
                        .add(builder::Input{static_cast<std::int64_t>(0), "Zero"})
                        .add(builder::Output{"O"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"ZS"}
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < out_shape.size(); ++idx)
                                            {
                                                if (idx == op().get_one_hot_axis())
                                                {
                                                    out = std::to_string(out_shape[idx]);
                                                }
                                                else
                                                {
                                                    out = "1";
                                                }
                                            }
                                        })
                                        .add_indices("d", 0, out_shape.size()))
                                .set(builder::ContractionInput{"Zero"}))
                        .add(builder::Elementwise{
                            "Idx", "index(ZS, " + std::to_string(op().get_one_hot_axis()) + ")"})
                        .add(builder::Elementwise{"IS", "reshape(I, " + in_reshape.str() + ")"})
                        .add(builder::Elementwise{"OV", "IS == Idx ? 1 : 0"})
                        .add(builder::Elementwise{"O",
                                                  tile_converter("OV", op().get_element_type())})
                        .finalize());
            }

            namespace
            {
                Impl<op::OneHot>::Registration register_one_hot;
            }
        }
    }
}
