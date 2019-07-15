//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/runtime/plaidml/plaidml_ops_replicate.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplReplicate, OpImpl<plaidml::op::Replicate>);
        }
    }
}

ngraph::runtime::plaidml::op::Replicate::Replicate(std::shared_ptr<Node> arg,
                                                   std::size_t replication_axis,
                                                   std::size_t replication_count)
    : Op{"Replicate", NodeVector{arg}}
    , m_replication_axes(arg->get_shape().size(), 1)
{
    m_replication_axes.at(replication_axis) = replication_count;
    constructor_validate_and_infer_types();
}

ngraph::runtime::plaidml::op::Replicate::Replicate(std::shared_ptr<Node> arg,
                                                   std::vector<std::size_t> replication_axes)
    : Op{"Replicate", NodeVector{arg}}
    , m_replication_axes(std::move(replication_axes))
{
    if (arg->get_shape().size() != m_replication_axes.size())
    {
        throw ngraph_error{"Replicate requires compatible axes dimensions"};
    }

    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::Replicate::validate_and_infer_types()
{
    std::shared_ptr<Node> arg = get_argument(0);
    Shape shape = arg->get_shape();
    for (auto rit = m_replication_axes.begin(), sit = shape.begin();
         rit != m_replication_axes.end();
         ++rit, ++sit)
    {
        *sit *= *rit;
    }
    set_output_type(0, arg->get_element_type(), shape);
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::Replicate::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error{"Replicate requires exactly one input"};
    }
    if (new_args.at(0)->get_shape().size() != m_replication_axes.size())
    {
        throw ngraph_error{"Replicate requires identical dimensions in inputs"};
    }
    return std::make_shared<Replicate>(new_args.at(0), m_replication_axes);
}

void ngraph::runtime::plaidml::ImplReplicate::Apply()
{
    check_inputs(1);
    check_outputs(1);

    const auto& axes = op().get_replication_axes();
    const auto& ishape = op().get_input_shape(0);
    set_output(
        start_tile_function()
            .add(builder::Input{op_input(0), "I"}.add_dims("D", 0, axes.size()))
            .add(builder::Output{"O"})
            .add(builder::UnaryContraction{"="}
                     .set(builder::ContractionOutput{"O"}
                              .add_dims([&](std::back_insert_iterator<std::list<std::string>> out) {
                                  for (std::size_t idx = 0; idx < axes.size(); ++idx)
                                  {
                                      std::string dsize = "D" + std::to_string(idx);
                                      if (axes.at(idx) != 1)
                                      {
                                          dsize = dsize + " * " + std::to_string(axes.at(idx));
                                      }
                                      out = dsize;
                                  }
                              })
                              .add_indices([&](
                                  std::back_insert_iterator<std::list<std::string>> out) {
                                  for (std::size_t idx = 0; idx < axes.size(); ++idx)
                                  {
                                      std::string didx = "d" + std::to_string(idx);
                                      if (axes.at(idx) != 1)
                                      {
                                          if (ishape.at(idx) == 1)
                                          {
                                              didx = didx + " + s" + std::to_string(idx);
                                          }
                                          else
                                          {
                                              didx = didx + " + (s" + std::to_string(idx) + " * " +
                                                     std::to_string(ishape.at(idx)) + ")";
                                          }
                                      }
                                      out = didx;
                                  }
                              }))
                     .set(builder::ContractionInput{"I"}.add_indices("d", 0, axes.size())))
            .finalize());
}
