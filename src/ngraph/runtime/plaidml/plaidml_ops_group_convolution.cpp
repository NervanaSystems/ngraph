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

#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplGroupConvolution, OpImpl<::ngraph::op::GroupConvolution>);
        }
    } // namespace runtime
} // namespace ngraph

// GroupConvolution implements a grouped convolution, with optional striding, padding, and dilation.
void ngraph::runtime::plaidml::ImplGroupConvolution::Apply()
{
    this->check_inputs(2);
    this->check_outputs(1);

    const auto& image = op_input(0);
    const auto& filter = op_input(1);

    auto rank = op().get_input_shape(0).size() - 2;
    const auto& groups = op().get_groups();
    const auto& padding_above = op().get_padding_above();
    const auto& padding_below = op().get_padding_below();
    const auto& strides = op().get_window_movement_strides();
    const auto& filter_dilation = op().get_window_dilation_strides();
    const auto& data_dilation = op().get_data_dilation_strides();

    const auto& grps =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(groups));
    const auto& dd0 =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(data_dilation[0]));
    const auto& dd1 =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(data_dilation[1]));
    const auto& fd0 =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(filter_dilation[0]));
    const auto& fd1 =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(filter_dilation[1]));
    const auto& pxb =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(padding_below[0]));
    const auto& pyb =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(padding_below[1]));
    const auto& pxa =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(padding_above[0]));
    const auto& pya =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(padding_above[1]));
    const auto& sx =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(strides[0]));
    const auto& sy =
        static_cast<::vertexai::plaidml::variable>(static_cast<const int64_t>(strides[1]));

    this->set_output(::vertexai::plaidml::function{R"(
            function (I[N, CI, XI0, XI1], F[CO, FCI, XF0, XF1], DD0, DD1, FD0, FD1, G, PXB, PYB, PXA, PYA, SX, SY) -> (O) {
                O[n, (CO/G) * g + co, x, y: N, CO, ((DD0 * (XI0 - 1) + PXA + PXB) - (FD0 * (XF0 - 1)) + SX) / SX, ((DD1 * (XI1 - 1) + PYA + PYB) - (FD1 * (XF1 - 1)) + SY) / SY] = 
                    +(I[n, (CI/G) * g + ci, (x + FD0 * xf0 - PXB)/DD0, (y + FD1 * xf1 - PYB)/DD1] * F[(CO/G) * g + co, ci, xf0, xf1]), co < CO/G;
            })"}(image, filter, dd0, dd1, fd0, fd1, grps, pxb, pyb, pxa, pya, sx, sy));
}
