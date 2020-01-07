//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
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
            class ReductionBase : public OpImpl<O>
            {
            public:
                void build_reduction(const char* agg_op);
            };

            template <typename O>
            void ReductionBase<O>::build_reduction(const char* agg_op)
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

            NGRAPH_PLAIDML_OP_CLASS(ImplAll, ReductionBase<op::All>);
            NGRAPH_PLAIDML_OP_CLASS(ImplAny, ReductionBase<op::Any>);
            NGRAPH_PLAIDML_OP_CLASS(ImplMax, ReductionBase<op::Max>);
            NGRAPH_PLAIDML_OP_CLASS(ImplMin, ReductionBase<op::Min>);
            NGRAPH_PLAIDML_OP_CLASS(ImplProduct, ReductionBase<op::Product>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSum, ReductionBase<op::Sum>);
        }
    }
}

// All reduces a tensor, taking the boolean minimum along the specified axes.
void ngraph::runtime::plaidml::ImplAll::Apply()
{
    build_reduction("<");
}

// Any reduces a tensor, taking the boolean maximum along the specified axes.
void ngraph::runtime::plaidml::ImplAny::Apply()
{
    build_reduction(">");
}

// Max reduces a tensor, taking the maximum along the specified axes.
void ngraph::runtime::plaidml::ImplMax::Apply()
{
    build_reduction(">");
}

// Min reduces a tensor, taking the minimum along the specified axes.
void ngraph::runtime::plaidml::ImplMin::Apply()
{
    build_reduction("<");
}

// Min reduces a tensor, taking the product along the specified axes.
void ngraph::runtime::plaidml::ImplProduct::Apply()
{
    build_reduction("*");
}

// Sum reduces a tensor, summing the specified axes.
void ngraph::runtime::plaidml::ImplSum::Apply()
{
    build_reduction("+");
}
