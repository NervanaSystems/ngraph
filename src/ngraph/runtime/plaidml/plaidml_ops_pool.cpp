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
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/runtime/plaidml/plaidml_convpool_formatter.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // AvgPool implements a batch average pooling operation.
            template <>
            void Impl<op::AvgPool>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                auto src_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& padding_above = op().get_padding_above();
                const auto& padding_below = op().get_padding_below();
                const auto& window_shape = op().get_window_shape();
                const auto& strides = op().get_window_movement_strides();
                const auto& include_padding = op().get_include_padding_in_avg_computation();

                ngraph::CoordinateDiff pad_above;
                ngraph::CoordinateDiff pad_below;
                for (const auto& pad : padding_above)
                {
                    pad_above.push_back(pad);
                }
                for (const auto& pad : padding_below)
                {
                    pad_below.push_back(pad);
                }

                // Overpadding occurs iff any padding value is >= its corresponding window shape.  If this
                // happens, we need to conditionally set the padded values to the operation default.

                bool overpad = false;
                for (std::size_t idx = 0; idx < src_dims; ++idx)
                {
                    auto shape = window_shape[idx];
                    if (shape <= padding_below[idx] || shape <= padding_above[idx])
                    {
                        overpad = true;
                        break;
                    }
                }

                if (overpad)
                {
                    throw std::runtime_error{
                        "The PlaidML nGraph backend does not support over-padded AvgPool "
                        "operations"};
                }

                ConvPoolFormatter cpf(src_dims,
                                      pad_below,
                                      pad_above,
                                      strides,
                                      window_shape,
                                      ConvPoolFormatter::OpType::AvgPool,
                                      ConvPoolFormatter::DerivType::None);

                vertexai::plaidml::variable one{static_cast<std::int64_t>(1)};

                auto f = start_tile_function();
                f.add(cpf.I_in_header(op_input()))
                    .add(builder::Input{one, "One"})
                    .add(cpf.O_out_header())
                    .add(cpf.Broadcast_Ones());
                if (include_padding)
                {
                    f.add(builder::Elementwise{"Count", std::to_string(shape_size(window_shape))});
                }
                else
                {
                    f.add(cpf.Count());
                }
                f.add(cpf.PoolContraction()).add(builder::Elementwise{"O", "S / Count"});

                set_output(f.finalize());
            }

            // MaxPool implements a batch max pooling operation.
            template <>
            void Impl<op::MaxPool>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                auto src_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& padding_above = op().get_padding_above();
                const auto& padding_below = op().get_padding_below();
                const auto& window_shape = op().get_window_shape();
                const auto& strides = op().get_window_movement_strides();
                ngraph::CoordinateDiff pad_above;
                ngraph::CoordinateDiff pad_below;
                for (const auto& pad : padding_above)
                {
                    pad_above.push_back(pad);
                }
                for (const auto& pad : padding_below)
                {
                    pad_below.push_back(pad);
                }

                NGRAPH_DEBUG << "MaxPool padding_below: " << padding_below;
                NGRAPH_DEBUG << "MaxPool padding_above: " << padding_above;
                NGRAPH_DEBUG << "MaxPool window_shape: " << window_shape;
                NGRAPH_DEBUG << "MaxPool window_movement_strides: " << strides;

                // Overpadding occurs iff any padding value is >= its corresponding window shape.  If this
                // happens, we need to conditionally set the padded values to the operation default.

                bool overpad = false;
                for (std::size_t idx = 0; idx < src_dims; ++idx)
                {
                    auto shape = window_shape[idx];
                    if (shape <= padding_below[idx] || shape <= padding_above[idx])
                    {
                        overpad = true;
                        break;
                    }
                }

                if (overpad)
                {
                    throw std::runtime_error{
                        "The PlaidML nGraph backend does not support over-padded MaxPool "
                        "operations"};
                }

                ConvPoolFormatter cpf(src_dims,
                                      pad_below,
                                      pad_above,
                                      strides,
                                      window_shape,
                                      ConvPoolFormatter::OpType::MaxPool,
                                      ConvPoolFormatter::DerivType::None);

                set_output(start_tile_function()
                               .add(cpf.I_in_header(op_input()))
                               .add(cpf.O_out_header())
                               .add(cpf.PoolContraction())
                               .finalize());
            }

            template <>
            void Impl<op::AvgPoolBackprop>::operator()()
            {
                check_inputs(1);
                check_outputs(1);

                auto src_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& forward_arg_shape = op().get_forward_arg_shape();
                const auto& padding_above = op().get_padding_above();
                const auto& padding_below = op().get_padding_below();
                const auto& window_shape = op().get_window_shape();
                const auto& strides = op().get_window_movement_strides();
                const auto& include_padding = op().get_include_padding_in_avg_computation();

                if (include_padding)
                {
                    throw std::runtime_error(
                        "Include padding in average not yet implemented in PlaidML");
                }

                ngraph::CoordinateDiff pad_above;
                ngraph::CoordinateDiff pad_below;
                for (const auto& pad : padding_above)
                {
                    pad_above.push_back(pad);
                }
                for (const auto& pad : padding_below)
                {
                    pad_below.push_back(pad);
                }

                // Overpadding occurs iff any padding value is >= its corresponding window shape.  If this
                // happens, we need to conditionally set the padded values to the operation default.

                bool overpad = false;
                for (std::size_t idx = 0; idx < src_dims; ++idx)
                {
                    auto shape = window_shape[idx];
                    if (shape <= padding_below[idx] || shape <= padding_above[idx])
                    {
                        overpad = true;
                        break;
                    }
                }

                if (overpad)
                {
                    throw std::runtime_error{
                        "The PlaidML nGraph backend does not support over-padded AvgPool "
                        "operations"};
                }

                ConvPoolFormatter cpf(src_dims,
                                      pad_below,
                                      pad_above,
                                      strides,
                                      window_shape,
                                      ConvPoolFormatter::OpType::AvgPool,
                                      ConvPoolFormatter::DerivType::Data);

                const auto& incoming_deriv = op_input();

                vertexai::plaidml::variable one{static_cast<std::int64_t>(1)};

                auto ret = start_tile_function();
                ret.add(cpf.O_in_header(incoming_deriv))
                    .add(builder::Input{one, "One"})
                    .add(builder::Output{"DI"});
                for (int i = 2; i < forward_arg_shape.size(); ++i)
                {
                    std::ostringstream s;
                    s << "XI" << i - 2;
                    ret.add(
                        builder::Input{static_cast<std::int64_t>(forward_arg_shape[i]), s.str()});
                }
                set_output(ret.add(cpf.Broadcast_Ones())
                               .add(cpf.Count())
                               .add(builder::Elementwise{"S", "DO / Count"})
                               .add(cpf.PoolContraction())
                               .finalize());
            }

            template <>
            void Impl<op::MaxPoolBackprop>::operator()()
            {
                check_inputs(2);
                check_outputs(1);

                auto src_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& padding_above = op().get_padding_above();
                const auto& padding_below = op().get_padding_below();
                const auto& window_shape = op().get_window_shape();
                const auto& strides = op().get_window_movement_strides();
                ngraph::CoordinateDiff pad_above;
                ngraph::CoordinateDiff pad_below;
                for (const auto& pad : padding_above)
                {
                    pad_above.push_back(pad);
                }
                for (const auto& pad : padding_below)
                {
                    pad_below.push_back(pad);
                }

                // Overpadding occurs iff any padding value is >= its corresponding window shape.  If this
                // happens, we need to conditionally set the padded values to the operation default.

                bool overpad = false;
                for (std::size_t idx = 0; idx < src_dims; ++idx)
                {
                    auto shape = window_shape[idx];
                    if (shape <= padding_below[idx] || shape <= padding_above[idx])
                    {
                        overpad = true;
                        break;
                    }
                }

                if (overpad)
                {
                    throw std::runtime_error{
                        "The PlaidML nGraph backend does not support over-padded MaxPool "
                        "operations"};
                }

                ConvPoolFormatter cpf(src_dims,
                                      pad_below,
                                      pad_above,
                                      strides,
                                      window_shape,
                                      ConvPoolFormatter::OpType::MaxPool,
                                      ConvPoolFormatter::DerivType::Data);

                const auto& input = op_input(0);
                const auto& incoming_deriv = op_input(1);

                set_output(start_tile_function()
                               .add(cpf.I_in_header(input))
                               .add(cpf.O_in_header(incoming_deriv))
                               .add(builder::Output{"DI"})
                               .add(cpf.PoolContraction())
                               .add(cpf.PoolDerivContraction())
                               .finalize());
            }

            namespace
            {
                Impl<op::AvgPool>::Registration register_avg_pool;
                Impl<op::MaxPool>::Registration register_max_pool;
                Impl<op::AvgPoolBackprop>::Registration register_avg_pool_backprop;
                Impl<op::MaxPoolBackprop>::Registration register_max_pool_backprop;
            }
        }
    }
}
