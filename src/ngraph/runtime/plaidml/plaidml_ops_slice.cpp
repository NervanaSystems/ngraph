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

#include "ngraph/log.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Slice takes a sub-slice of a tensor.
            template <>
            void Impl<op::Slice>::operator()()
            {
                check_inputs(1);
                check_outputs(1);
                NGRAPH_DEBUG << "Slice: low: " << op().get_lower_bounds();
                NGRAPH_DEBUG << "Slice high: " << op().get_upper_bounds();
                NGRAPH_DEBUG << "Slice stride: " << op().get_strides();
                const auto& shape = op().get_inputs()[0].get_shape();
                auto dim_limit = shape.size();
                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(), "I"}.add_dims("ID", 0, dim_limit))
                        .add(builder::Output{"O"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"O"}
                                        .add_indices("od", 0, dim_limit)
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                            {
                                                std::ostringstream s;
                                                std::size_t stride = op().get_strides()[idx];
                                                std::ptrdiff_t trim_count =
                                                    op().get_lower_bounds()[idx] +
                                                    (shape[idx] - op().get_upper_bounds()[idx]) +
                                                    1 - stride;
                                                if ((stride != 1) && trim_count)
                                                {
                                                    s << "(";
                                                }
                                                s << "ID" << idx;
                                                if (0 < trim_count)
                                                {
                                                    s << " - " << trim_count;
                                                }
                                                if (trim_count < 0)
                                                {
                                                    s << " + " << -trim_count;
                                                }
                                                if ((stride != 1) && trim_count)
                                                {
                                                    s << ")";
                                                }
                                                if (stride != 1)
                                                {
                                                    s << " / " << stride;
                                                }
                                                out = s.str();
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}.add_indices(
                                    [&](std::back_insert_iterator<std::list<std::string>> out) {
                                        for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                        {
                                            std::ostringstream s;
                                            std::size_t stride = op().get_strides()[idx];
                                            std::size_t offset = op().get_lower_bounds()[idx];
                                            if ((stride != 1) && offset)
                                            {
                                                s << "(";
                                            }
                                            s << "od" << idx;
                                            if (stride != 1)
                                            {
                                                s << " * " << stride;
                                            }
                                            if ((stride != 1) && offset)
                                            {
                                                s << ")";
                                            }
                                            if (offset)
                                            {
                                                s << " + " << offset;
                                            }
                                            out = s.str();
                                        }
                                    })))
                        .finalize());
            }

            namespace
            {
                Impl<op::Slice>::Registration register_slice;
            }
        }
    }
}
