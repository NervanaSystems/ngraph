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
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Broadcast broadcasts a tensor to a wider shape.
            template <>
            void Impl<op::Broadcast>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                auto in_dim_limit = op().get_inputs()[0].get_shape().size();
                auto out_dim_limit = op().get_broadcast_shape().size();

                NGRAPH_DEBUG << "Broadcast in_dim_limit: " << in_dim_limit
                             << " out_dim_limit:" << out_dim_limit;
                NGRAPH_DEBUG << "Broadcast axes: " << op().get_broadcast_axes();
                NGRAPH_DEBUG << "Broadcast input shape: " << op().get_input_shape(0);
                NGRAPH_DEBUG << "Broadcast output shape: " << op().get_broadcast_shape();

                auto input_didx = in_dim_limit;
                std::vector<std::size_t> out_didxs;
                for (std::size_t idx = 0; idx < out_dim_limit; ++idx)
                {
                    if (!op().get_broadcast_axes().count(idx))
                    {
                        out_didxs.push_back(out_dim_limit - idx - 1);
                    }
                }
                set_output(
                    start_tile_function()
                        .add(builder::Input{op_input(0), "I"}.add_rdims("D", in_dim_limit, 0))
                        .add(builder::Output{"O"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"O"}
                                        .add_rindices("o", out_dim_limit, 0)
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < out_dim_limit; ++idx)
                                            {
                                                if (op().get_broadcast_axes().count(idx))
                                                {
                                                    out = std::to_string(
                                                        op().get_broadcast_shape()[idx]);
                                                }
                                                else
                                                {
                                                    out = "D" + std::to_string(--input_didx);
                                                }
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}.add_indices(
                                    [&](std::back_insert_iterator<std::list<std::string>> out) {
                                        for (std::size_t idx = 0; idx < in_dim_limit; ++idx)
                                        {
                                            out = "o" + std::to_string(out_didxs[idx]);
                                        }
                                    })))
                        .finalize());
            }

            // Constant fills in a tensor constant.
            template <>
            void Impl<op::Constant>::operator()()
            {
                check_inputs(0);
                check_outputs(1);

                bool output_to_result = false;
                for (const std::shared_ptr<Node>& node : op().get_users())
                {
                    if (dynamic_cast<op::Result*>(node.get()))
                    {
                        output_to_result = true;
                        break;
                    }
                }

                if (!op().get_shape().size() && !output_to_result)
                {
                    switch (to_plaidml(op().get_element_type()))
                    {
                    case PLAIDML_DATA_BOOLEAN:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const char*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_INT8:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::int8_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_INT16:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::int16_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_INT32:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::int32_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_INT64:
                        set_output(*static_cast<const std::int64_t*>(op().get_data_ptr()));
                        return;
                    case PLAIDML_DATA_UINT8:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::uint8_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_UINT16:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::uint16_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_UINT32:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::uint32_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_UINT64:
                        set_output(static_cast<std::int64_t>(
                            *static_cast<const std::uint64_t*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_FLOAT16:
                        set_output(static_cast<double>(
                            static_cast<float>(*static_cast<const half*>(op().get_data_ptr()))));
                        return;
                    case PLAIDML_DATA_FLOAT32:
                        set_output(
                            static_cast<double>(*static_cast<const float*>(op().get_data_ptr())));
                        return;
                    case PLAIDML_DATA_FLOAT64:
                        set_output(
                            static_cast<double>(*static_cast<const double*>(op().get_data_ptr())));
                        return;
                    default: break;
                    }
                }

                auto tensor = build()->config->dev->allocate(
                    to_plaidml(build()->config->ctx, op().get_element_type(), op().get_shape()));

                {
                    vp::mapping<char> mp = tensor.map(vp::map_for_write);
                    const char* src = static_cast<const char*>(op().get_data_ptr());
                    char* dest = mp.raw();
                    std::copy(src, src + tensor.get_shape().buffer_size(), dest);
                }

                set_output(tensor);
            }

            // GetOutputElement pipes one of its N inputs to its output.
            template <>
            void Impl<op::GetOutputElement>::operator()()
            {
                check_inputs_ge(op().get_n() + 1);
                check_outputs(1);

                set_output(op_input(op().get_n()));
            }

            // Pad adds interior and exterior padding to a tensor.
            template <>
            void Impl<op::Pad>::operator()()
            {
                check_inputs(2);
                check_outputs(1);

                auto tensor = op_input(0);
                auto value = op_input(1);

                // For padding, we construct two intermediate tensors: the first is the input tensor expanded by
                // the requisite padding (with zeros in all padded locations), and the second is a boolean
                // tensor expanded the same way, but with true at the source locations and false at the padded
                // locations.  We then combine these elementwise using a trinary condition, with the pad value
                // being used everywhere the boolean intermediate is false.

                // It's a little wasteful, but it expresses the logic correctly, and doesn't take long to run;
                // the runtime is also free to optimize it through combining the intermediate contractions.

                NGRAPH_DEBUG << "Pad below: " << op().get_padding_below();
                NGRAPH_DEBUG << "Pad above: " << op().get_padding_above();
                NGRAPH_DEBUG << "Pad interior: " << op().get_padding_interior();
                NGRAPH_DEBUG << "Pad input dims: " << op().get_input_shape(0);
                NGRAPH_DEBUG << "Pad output dims: " << op().get_shape();

                auto dim_limit = op().get_shape().size();

                bool any_zero_dims = false;
                for (auto sz : op().get_input_shape(0))
                {
                    if (!sz)
                    {
                        any_zero_dims = true;
                        break;
                    }
                }

                auto out_dsize = [&](std::size_t idx) {
                    std::ostringstream s;
                    std::size_t total_pad =
                        op().get_padding_below().at(idx) + op().get_padding_above().at(idx);
                    std::size_t in_dsize = op().get_input_shape(0).at(idx);
                    if (in_dsize)
                    {
                        total_pad += op().get_padding_interior().at(idx) * (in_dsize - 1);
                    }
                    if (!any_zero_dims)
                    {
                        s << "DI" << idx + 1;
                        if (total_pad)
                        {
                            s << " + " << total_pad;
                        }
                    }
                    else
                    {
                        s << total_pad + in_dsize;
                    }
                    return s.str();
                };

                auto out_didx = [&](std::size_t idx) {
                    std::ostringstream s;
                    auto below = op().get_padding_below().at(idx);
                    if (below)
                    {
                        s << below << " + ";
                    }
                    auto interior = op().get_padding_interior().at(idx) + 1;
                    if (interior != 1)
                    {
                        s << "(d" << idx + 1 << " * " << interior << ")";
                    }
                    else
                    {
                        s << "d" << idx + 1;
                    }
                    return s.str();
                };

                auto flag_constraints = [&](std::size_t idx) {
                    std::ostringstream s;
                    s << "d" << idx + 1 << " < DI" << idx + 1;
                    return s.str();
                };

                auto f = start_tile_function();

                f.add(builder::Input{op_input(1), "V"}).add(builder::Output{"O"});

                if (!any_zero_dims)
                {
                    f.add(builder::Input{op_input(0), "I"}.add_dims("DI", 1, dim_limit + 1))
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"P"}
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                            {
                                                out = out_didx(idx);
                                            }
                                        })
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                            {
                                                out = out_dsize(idx);
                                            }
                                        }))
                                .set(builder::ContractionInput{"I"}.add_indices(
                                    "d", 1, dim_limit + 1)))
                        .add(builder::Elementwise{"T", "1"})
                        .add(
                            builder::UnaryContraction{"="}
                                .set(
                                    builder::ContractionOutput{"F"}
                                        .add_indices([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                            {
                                                out = out_didx(idx);
                                            }
                                        })
                                        .add_dims([&](
                                            std::back_insert_iterator<std::list<std::string>> out) {
                                            for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                            {
                                                out = out_dsize(idx);
                                            }
                                        }))
                                .set(builder::ContractionInput{"T"})
                                .add_constraints(
                                    [&](std::back_insert_iterator<std::list<std::string>> out) {
                                        for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                        {
                                            out = flag_constraints(idx);
                                        }
                                    }))
                        .add(builder::Elementwise{"O", "F ? P : V"});
                }
                else
                {
                    f.add(builder::UnaryContraction{"="}
                              .set(builder::ContractionOutput{"O"}
                                       .add_indices("d", 0, dim_limit)
                                       .add_dims([&](
                                           std::back_insert_iterator<std::list<std::string>> out) {
                                           for (std::size_t idx = 0; idx < dim_limit; ++idx)
                                           {
                                               out = out_dsize(idx);
                                           }
                                       }))
                              .set(builder::ContractionInput{"V"}));
                }

                set_output(f.finalize());
            }

            // Reshape reshapes an input tensor.
            template <>
            void Impl<op::Reshape>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                // The reshape operation doesn't just describe a new way of looking at an input tensor; it can
                // optionally rearrange the elements of the input tensor.

                auto src = op_input(0);
                auto dim_limit = op().get_inputs()[0].get_shape().size();

                if (!dim_limit)
                {
                    // This reshape is being used to create a tensor from a scalar.  PlaidML's reshape()
                    // operation requires a tensor input (as of this writing), so instead of a reshape(), we'll
                    // just use a contraction to build the tensor.
                    auto& out_shape = op().get_shape();
                    set_output(
                        start_tile_function()
                            .add(builder::Input{src, "I"})
                            .add(builder::Output{"O"})
                            .add(builder::UnaryContraction{"="}
                                     .set(builder::ContractionOutput{"O"}
                                              .add_indices("d", 0, out_shape.size())
                                              .add_dims([&](
                                                  std::back_insert_iterator<std::list<std::string>>
                                                      out) {
                                                  std::transform(out_shape.begin(),
                                                                 out_shape.end(),
                                                                 out,
                                                                 [](std::size_t sz) {
                                                                     return std::to_string(sz);
                                                                 });
                                              }))
                                     .set(builder::ContractionInput{"I"}))
                            .finalize());
                    return;
                }

                std::size_t dim_idx = 0;
                auto input_order = op().get_input_order();
                for (std::size_t src_idx : op().get_input_order())
                {
                    if (src_idx != dim_idx++)
                    {
                        // This reshape operation doesn't just describe a new way of looking at an input tensor;
                        // it's also rearranging the elements of the input tensor.  This is pretty easy to
                        // handle with a contraction.

                        src = start_tile_function()
                                  .add(builder::Input{src, "I"}.add_dims("D", 1, dim_limit + 1))
                                  .add(builder::Output{"O"})
                                  .add(builder::UnaryContraction{"="}
                                           .set(builder::ContractionOutput{"O"}
                                                    .add_indices([&](std::back_insert_iterator<
                                                                     std::list<std::string>> out) {
                                                        for (std::size_t idx = 0; idx < dim_limit;
                                                             ++idx)
                                                        {
                                                            out = "d" + std::to_string(
                                                                            input_order[idx] + 1);
                                                        }
                                                    })
                                                    .add_dims([&](std::back_insert_iterator<
                                                                  std::list<std::string>> out) {
                                                        for (std::size_t idx = 0; idx < dim_limit;
                                                             ++idx)
                                                        {
                                                            out = "D" + std::to_string(
                                                                            input_order[idx] + 1);
                                                        }
                                                    }))
                                           .set(builder::ContractionInput{"I"}.add_indices(
                                               "d", 1, dim_limit + 1)))
                                  .finalize();
                        break;
                    }
                }

                std::ostringstream reshape_expr;
                reshape_expr << "reshape(I";
                for (std::size_t dsize : op().get_output_shape())
                {
                    reshape_expr << ", " << dsize;
                }
                reshape_expr << ")";

                set_output(start_tile_function()
                               .add(builder::Input{src, "I"})
                               .add(builder::Output{"O"})
                               .add(builder::Elementwise("O", reshape_expr.str()))
                               .finalize());
            }

            // Select conditionally selects elements from input tensors.
            template <>
            void Impl<op::Select>::operator()()
            {
                check_inputs(3);
                check_outputs(1);

                set_output(start_tile_function()
                               .add(builder::Input{op_input(0), "C"})
                               .add(builder::Input{op_input(1), "T"})
                               .add(builder::Input{op_input(2), "F"})
                               .add(builder::Output{"O"})
                               .add(builder::Elementwise{"O", "C ? T : F"})
                               .finalize());
            }

            // Used by nGraph for bprop graph generation, no-op as a kernel
            template <>
            void Impl<op::StopGradient>::operator()()
            {
                set_output(start_tile_function()
                               .add(builder::Output{"O"})
                               .add(builder::Elementwise{"O", "0"})
                               .finalize());
            }

            namespace
            {
                Impl<op::Broadcast>::Registration register_broadcast;
                Impl<op::Constant>::Registration register_constant;
                Impl<op::GetOutputElement>::Registration register_get_output_element;
                Impl<op::Pad>::Registration register_pad;
                Impl<op::Reshape>::Registration register_reshape;
                Impl<op::Select>::Registration register_select;
                Impl<op::StopGradient>::Registration register_stop_gradient;
            }
        }
    }
}
