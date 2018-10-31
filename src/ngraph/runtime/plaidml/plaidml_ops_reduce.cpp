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
#include "ngraph/op/reduce.hpp"
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
            struct ParentImpl<op::Reduce>
            {
                using Type = ReductionImpl<op::Reduce>;
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

            // Reduce reduces a tensor with an arbitrary user-supplied reduction operation.
            template <>
            void Impl<op::Reduce>::operator()()
            {
                check_inputs(2);
                check_outputs(1);

                // TODO: Special case known-easy reductions.

                // To support arbitrary reduction operations, we take advantage of the fact that in nGraph, we
                // have concrete dimension sizes.  We start with the initial tensor (argument 1), construct N
                // slices of tensor 0 (where N == the product of the sizes of the axes to reduce), and
                // repeatedly apply the supplied aggregation function to them.
                //
                // This is somewhat inefficient, but works.
                const Shape& input_shape = op().get_input_shape(0);
                auto dim_limit = input_shape.size();
                Shape reduction_shape;
                for (std::size_t axis_idx = 0; axis_idx < input_shape.size(); ++axis_idx)
                {
                    if (op().get_reduction_axes().count(axis_idx))
                    {
                        reduction_shape.emplace_back(input_shape[axis_idx]);
                    }
                }
                std::size_t agg_dim_limit = dim_limit - reduction_shape.size();

                vp::function agg_fn;
                {
                    Build b;
                    b.io_dim_override = true;
                    b.io_dim_override_count = agg_dim_limit;
                    build()->compiler->build(op().get_functions()[0], &b);
                    agg_fn = b.composer;
                }

                vp::variable input = op_input(0);

                // Note that we need to explicitly broadcast the 0-dimensional base result to match the
                // aggregation dimension count.
                vp::variable result =
                    start_tile_function()
                        .add(builder::Input{op_input(1), "I"})
                        .add(builder::Output{"O"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"O"}
                                        .add_indices("d", 0, agg_dim_limit)
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (auto idx = 0; idx < agg_dim_limit; ++idx)
                                            {
                                                out = "1";
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}))
                        .finalize();

                CoordinateTransform reduction_coords{reduction_shape};
                for (const Coordinate& coordinate : reduction_coords)
                {
                    result = agg_fn(
                        result,
                        start_tile_function()
                            .add(builder::Input{input, "I"}.add_dims("D", 0, dim_limit))
                            .add(builder::Output{"O"})
                            .add(builder::UnaryContraction{"="}
                                     .set(builder::ContractionOutput{"O"}
                                              .add_indices([&](
                                                  std::back_insert_iterator<std::list<std::string>>
                                                      out) {
                                                  for (std::size_t idx = 0;
                                                       idx < input_shape.size();
                                                       ++idx)
                                                  {
                                                      if (!op().get_reduction_axes().count(idx))
                                                      {
                                                          out = "d" + std::to_string(idx);
                                                      }
                                                  }
                                              })
                                              .add_dims([&](
                                                  std::back_insert_iterator<std::list<std::string>>
                                                      out) {
                                                  for (std::size_t idx = 0;
                                                       idx < input_shape.size();
                                                       ++idx)
                                                  {
                                                      if (!op().get_reduction_axes().count(idx))
                                                      {
                                                          out = "D" + std::to_string(idx);
                                                      }
                                                  }
                                              }))
                                     .set(builder::ContractionInput{"I"}.add_indices([&](
                                         std::back_insert_iterator<std::list<std::string>> out) {
                                         for (std::size_t idx = 0; idx < input_shape.size(); ++idx)
                                         {
                                             std::size_t cidx = 0;
                                             if (!op().get_reduction_axes().count(idx))
                                             {
                                                 out = "d" + std::to_string(idx);
                                             }
                                             else
                                             {
                                                 out = std::to_string(coordinate[cidx++]);
                                             }
                                         }
                                     })))
                            .finalize());
                }

                set_output(result);
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
                Impl<op::Reduce>::Registration register_reduce;
                Impl<op::Sum>::Registration register_sum;
            }
        }
    }
}
