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

#include "ngraph/runtime/plaidml/plaidml_ops_winograd.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplWinograd, OpImpl<plaidml::op::Winograd>);
        }
    }
}

const std::string ngraph::runtime::plaidml::op::Winograd::type_name{"Winograd"};

ngraph::runtime::plaidml::op::Winograd::Winograd(std::shared_ptr<plaidml::op::Convolution> conv,
                                                 const OutputVector& args)
    : Op{args}
    , m_conv{std::move(conv)}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::Winograd::validate_and_infer_types()
{
    set_output_type(0, m_conv->get_element_type(), m_conv->get_output_partial_shape(0));
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::Winograd::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error{"Winograd requires five inputs (data, filters, A, B, and G)"};
    }
    return std::make_shared<Winograd>(m_conv, as_output_vector(new_args));
}

void ngraph::runtime::plaidml::ImplWinograd::Apply()
{
    check_inputs(5);
    check_outputs(1);

    const auto& data_shape = op().get_input_shape(0);
    const auto& filters_shape = op().get_input_shape(1);
    const auto& padding_above = op().get_conv()->get_src()->get_padding_above();
    const auto& padding_below = op().get_conv()->get_src()->get_padding_below();

    vp::variable xo(static_cast<int64_t>(data_shape.at(1) + padding_below.at(0) +
                                         padding_above.at(0) - filters_shape.at(0) + 1));
    vp::variable yo(static_cast<int64_t>(data_shape.at(2) + padding_below.at(1) +
                                         padding_above.at(1) - filters_shape.at(1) + 1));
    vp::variable xp(static_cast<int64_t>(padding_below.at(0)));
    vp::variable yp(static_cast<int64_t>(padding_below.at(1)));

    set_output(vp::function{R"(
            function (I[N, X, Y, CI], K[S, S, CI, CO], A[BI, BO], B[BI, BI], G[BI, S], XO, YO, XP, YP) -> (O) {
                Assert = assert_winograd_valid(BI - CI + 1 == BO);
                XB = (XO + BO - 1) / BO;
                YB = (YO + BO - 1) / BO;
                U1[i, j, ci, co : BI, S, CI, CO] = +(G[i, k] * K[k, j, ci, co]);
                U[i, j, ci, co : BI, BI, CI, CO] = +(U1[i, k, ci, co] * G[j, k]);
                V1[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(B[k, i] * I[n, BO*x + k - XP, BO*y + j - YP, ci]);
                V[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(V1[n, i, k, x, y, ci] * B[k, j]);
                M[n, i, j, x, y, co : N, BI, BI, XB, YB, CO] = +(V[n, i, j, x, y, ci] * U[i, j, ci, co]);
                O1[n, i, j, x, y, co : N, BO, BI, XB, YB, CO] = +(A[k, i] * M[n, k, j, x, y, co]);
                O[n, BO*x + i, BO*y + j, co : N, XO, YO, CO] = +(O1[n, i, k, x, y, co] * A[k, j]) no_defract;
            })"}(op_input(0), op_input(1), op_input(2), op_input(3), op_input(4), xo, yo, xp, yp));
}
