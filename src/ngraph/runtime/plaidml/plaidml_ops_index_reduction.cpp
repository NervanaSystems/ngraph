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

#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            template <typename O>
            class IndexReductionImpl : public BaseImpl<O>
            {
            public:
                IndexReductionImpl(Build* build, const O& op)
                    : BaseImpl<O>{build, op}
                {
                }

                void build_index_reduction(const char* agg_op);
            };

            template <typename O>
            void IndexReductionImpl<O>::build_index_reduction(const char* agg_op)
            {
                this->check_inputs(1);
                this->check_outputs(1);

                auto dim_limit = this->op().get_inputs()[0].get_shape().size();

                auto reduction_axis_str = std::to_string(this->op().get_reduction_axis());

                this->set_output(
                    this->start_tile_function()
                        .add(builder::Input{this->op_input(), "I"}.add_dims("D", 0, dim_limit))
                        .add(builder::Output{"O"})
                        .add( // Compute the maxes along the specified axis in the input
                            builder::UnaryContraction{agg_op}
                                .set(
                                    builder::ContractionOutput{"SelVal"}
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (auto idx = 0; idx < dim_limit; ++idx)
                                            {
                                                out =
                                                    (idx == this->op().get_reduction_axis() ? "rd"
                                                                                            : "d") +
                                                    std::to_string(idx);
                                            }
                                        })
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (auto idx = 0; idx < dim_limit; ++idx)
                                            {
                                                if (idx == this->op().get_reduction_axis())
                                                {
                                                    out = "1";
                                                }
                                                else
                                                {
                                                    out = "D" + std::to_string(idx);
                                                }
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}.add_indices("d", 0, dim_limit)))
                        .add( // Compare the input against the (broadcasted) max values, and select the indices
                            // where the max val occurs
                            builder::Elementwise{"SelValIdxs",
                                                 "I == SelVal ? index(I, " + reduction_axis_str +
                                                     ") : D" + reduction_axis_str})
                        .add( // Select the maximum index
                            builder::UnaryContraction{"<"}
                                .set(
                                    builder::ContractionOutput{"SelIdx"}
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (auto idx = 0; idx < dim_limit; ++idx)
                                            {
                                                if (idx != this->op().get_reduction_axis())
                                                {
                                                    out = "d" + std::to_string(idx);
                                                }
                                            }
                                        })
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (auto idx = 0; idx < dim_limit; ++idx)
                                            {
                                                if (idx != this->op().get_reduction_axis())
                                                {
                                                    out = "D" + std::to_string(idx);
                                                }
                                            }
                                        }))
                                .set(builder::ContractionInput{"SelValIdxs"}.add_indices(
                                    "d", 0, dim_limit)))
                        .add( // Convert to the requested output element type (if any)
                            builder::Elementwise{
                                "O", tile_converter("SelIdx", this->op().get_index_element_type())})
                        .finalize());
            }

            template <>
            struct ParentImpl<op::ArgMax>
            {
                using Type = IndexReductionImpl<op::ArgMax>;
            };

            template <>
            struct ParentImpl<op::ArgMin>
            {
                using Type = IndexReductionImpl<op::ArgMin>;
            };

            // ArgMax computes the maximum index along a tensor axis.
            template <>
            void Impl<op::ArgMax>::operator()()
            {
                build_index_reduction(">");
            }

            // ArgMin computes the minimum index along a tensor axis.
            template <>
            void Impl<op::ArgMin>::operator()()
            {
                build_index_reduction("<");
            }

            namespace
            {
                Impl<op::ArgMax>::Registration register_argmax;
                Impl<op::ArgMin>::Registration register_argmin;
            }
        }
    }
}
