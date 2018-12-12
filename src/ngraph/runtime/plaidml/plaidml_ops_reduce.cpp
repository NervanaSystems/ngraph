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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiler.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            template <typename O>
            class ReductionImpl : public BaseImpl<O>
            {
            public:
                ReductionImpl(Build* build, const O& op)
                    : BaseImpl<O>{build, op}
                {
                }

                void build_reduction(const char* agg_op);
            };

            template <typename O>
            void ReductionImpl<O>::build_reduction(const char* agg_op)
            {
                this->check_inputs(1);
                this->check_outputs(1);

                auto in_shape = this->op().get_input_shape(0);
                auto in_dim_limit = in_shape.size();

                std::vector<std::size_t> out_idxs;
                for (std::size_t in_idx = 0; in_idx < in_dim_limit; ++in_idx)
                {
                    if (!this->op().get_reduction_axes().count(in_idx))
                    {
                        out_idxs.push_back(in_idx);
                    }
                }

                this->set_output(
                    this->start_tile_function()
                        .add(builder::Output{"O"})

                        .add(builder::Input{this->op_input(0), "I"}.add_dims(
                            "D", 1, in_dim_limit + 1))
                        .add(
                            builder::UnaryContraction{agg_op}
                                .set(
                                    builder::ContractionOutput{"O"}
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < out_idxs.size(); ++idx)
                                            {
                                                out = "d" + std::to_string(out_idxs[idx] + 1);
                                            }
                                        })
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < out_idxs.size(); ++idx)
                                            {
                                                out = "D" + std::to_string(out_idxs[idx] + 1);
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}.add_indices(
                                    "d", 1, in_dim_limit + 1)))
                        .finalize());
            }

            template <>
            struct ParentImpl<op::Max>
            {
                using Type = ReductionImpl<op::Max>;
            };

            template <>
            struct ParentImpl<op::Min>
            {
                using Type = ReductionImpl<op::Min>;
            };

            template <>
            struct ParentImpl<op::Product>
            {
                using Type = ReductionImpl<op::Product>;
            };

            template <>
            struct ParentImpl<op::Sum>
            {
                using Type = ReductionImpl<op::Sum>;
            };

            // Max reduces a tensor, taking the maximum along the specified axes.
            template <>
            void Impl<op::Max>::operator()()
            {
                build_reduction(">");
            }

            // Min reduces a tensor, taking the minimum along the specified axes.
            template <>
            void Impl<op::Min>::operator()()
            {
                build_reduction("<");
            }

            // Min reduces a tensor, taking the product along the specified axes.
            template <>
            void Impl<op::Product>::operator()()
            {
                build_reduction("*");
            }

            // Sum reduces a tensor, summing the specified axes.
            template <>
            void Impl<op::Sum>::operator()()
            {
                build_reduction("+");
            }

            namespace
            {
                Impl<op::Max>::Registration register_max;
                Impl<op::Min>::Registration register_min;
                Impl<op::Product>::Registration register_product;
                Impl<op::Sum>::Registration register_sum;
            }
        }
    }
}
