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

#include "ngraph/runtime/plaidml/plaidml_convpool_formatter.hpp"

ngraph::runtime::plaidml::ConvPoolFormatter::ConvPoolFormatter(
    std::size_t rank,
    const ngraph::CoordinateDiff& pad_below,
    const ngraph::CoordinateDiff& pad_above,
    const ngraph::Strides& strides,
    const ngraph::Strides& filter_dilation,
    const ngraph::Strides& data_dilation,
    ConvPoolFormatter::OpType op,
    ConvPoolFormatter::DerivType deriv,
    const ngraph::Shape& deriv_output_shape)
    : m_rank{rank}
    , m_pad_below{pad_below}
    , m_pad_above{pad_above}
    , m_strides{strides}
    , m_filter_dilation{filter_dilation}
    , m_data_dilation{data_dilation}
    , m_op{op}
    , m_deriv{deriv}
{
    m_window_shape = Shape(rank, 0); // Not used for convolutions
    if (m_op != OpType::Conv)
    {
        throw std::runtime_error{"Using conv-style ctor for pool"};
    }
    if (m_pad_below.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in pad_below ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_pad_below.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_pad_above.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in pad_above ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_pad_above.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_strides.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in strides ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_strides.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_filter_dilation.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in filter dilation ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_filter_dilation.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_data_dilation.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in data dilation ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_data_dilation.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_deriv == DerivType::None && !deriv_output_shape.empty())
    {
        throw std::runtime_error{"Forward pass given derivative shape"};
    }
    if (m_deriv == DerivType::Filter)
    {
        m_filters_shape = deriv_output_shape;
        if (m_filters_shape.size() != rank + 2)
        {
            std::ostringstream msg;
            msg << "Rank mismatch in filter shape ";
            msg << "(expected length " << rank + 2 << " to match rank " << rank;
            msg << " but received length " << m_filters_shape.size() << ")";
            throw std::runtime_error{msg.str()};
        }
    }
    if (m_deriv == DerivType::Data)
    {
        m_data_batch_shape = deriv_output_shape;
        if (m_data_batch_shape.size() != rank + 2)
        {
            std::ostringstream msg;
            msg << "Rank mismatch in data batch shape ";
            msg << "(expected length " << rank + 2 << " to match rank " << rank;
            msg << " but received length " << m_data_batch_shape.size() << ")";
            throw std::runtime_error{msg.str()};
        }
    }
}

ngraph::runtime::plaidml::ConvPoolFormatter::ConvPoolFormatter(
    std::size_t rank,
    const ngraph::CoordinateDiff& pad_below,
    const ngraph::CoordinateDiff& pad_above,
    const ngraph::Strides& strides,
    const ngraph::Shape& window_shape,
    ConvPoolFormatter::OpType op,
    ConvPoolFormatter::DerivType deriv)
    : m_rank{rank}
    , m_pad_below{pad_below}
    , m_pad_above{pad_above}
    , m_strides{strides}
    , m_window_shape{window_shape}
    , m_op{op}
    , m_deriv{deriv}
{
    m_filter_dilation = ngraph::Strides(rank, 1); // Not used for pools
    m_data_dilation = ngraph::Strides(rank, 1);   // Nos used for pools
    if (m_op == OpType::Conv)
    {
        throw std::runtime_error{"Using pool-style ctor for conv"};
    }
    if (m_deriv == DerivType::Filter)
    {
        throw std::runtime_error{"Asking for filter deriv for pool"};
    }
    if (m_pad_below.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in pad_below ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_pad_below.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_pad_above.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in pad_above ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_pad_above.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_strides.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in strides ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_strides.size() << ")";
        throw std::runtime_error{msg.str()};
    }
    if (m_window_shape.size() != rank)
    {
        std::ostringstream msg;
        msg << "Rank mismatch in window shape ";
        msg << "(expected length " << rank << " to match rank " << rank;
        msg << " but received length " << m_filter_dilation.size() << ")";
        throw std::runtime_error{msg.str()};
    }
}

ngraph::runtime::plaidml::builder::Input
    ngraph::runtime::plaidml::ConvPoolFormatter::F_in_header(vertexai::plaidml::variable var)
{
    if (m_op != OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct filter F for pooling operation"};
    }
    if (m_deriv == DerivType::Filter)
    {
        throw std::runtime_error{"Asked to construct F as input when computing its gradient"};
    }
    builder::Input ret{var, F()};
    ret.add_dims({CO(), CI()});
    for (const auto& XFi : XFs())
    {
        ret.add_dims({XFi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::Input
    ngraph::runtime::plaidml::ConvPoolFormatter::I_in_header(vertexai::plaidml::variable var)
{
    if (m_deriv == DerivType::Data && m_op == OpType::Conv)
    {
        throw std::runtime_error{
            "Asked to construct I as input to convolution when computing its gradient"};
    }
    builder::Input ret{var, "I"};
    ret.add_dims({N()});
    if (m_op == OpType::Conv)
    {
        ret.add_dims({CI()});
    }
    else
    {
        ret.add_dims({C()});
    }
    for (const auto& XIi : XIs())
    {
        ret.add_dims({XIi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::Input
    ngraph::runtime::plaidml::ConvPoolFormatter::O_in_header(vertexai::plaidml::variable var)
{
    if (m_deriv == DerivType::None)
    {
        throw std::runtime_error{"Asked to construct O as input in forward pass"};
    }
    builder::Input ret{var, O()};
    ret.add_dims({N()});
    if (m_op == OpType::Conv)
    {
        ret.add_dims({CO()});
    }
    else
    {
        ret.add_dims({C()});
    }
    for (const auto& XOi : XOs())
    {
        ret.add_dims({XOi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::Output
    ngraph::runtime::plaidml::ConvPoolFormatter::F_out_header()
{
    if (m_op != OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct filter F for pooling operation"};
    }
    if (m_deriv != DerivType::Filter)
    {
        throw std::runtime_error{"Asked for output F when not finding gradient w.r.t. F"};
    }
    return builder::Output{F()};
}

ngraph::runtime::plaidml::builder::Output
    ngraph::runtime::plaidml::ConvPoolFormatter::I_out_header()
{
    if (m_deriv != DerivType::Data)
    {
        throw std::runtime_error{"Asked to construct I as output in forward pass"};
    }
    if (m_op == OpType::Conv)
    {
        return builder::Output{"DI"};
    }
    else
    {
        // TODO: Confirm correct for AvgPool as well
        return builder::Output{"I"};
    }
}

ngraph::runtime::plaidml::builder::Output
    ngraph::runtime::plaidml::ConvPoolFormatter::O_out_header()
{
    if (m_deriv != DerivType::None)
    {
        throw std::runtime_error{"Asked to construct O as output in gradient pass"};
    }
    return builder::Output{O()};
}

ngraph::runtime::plaidml::builder::ContractionOutput
    ngraph::runtime::plaidml::ConvPoolFormatter::F_out_body()
{
    if (m_op != OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct filter F for pooling operation"};
    }
    if (m_deriv != DerivType::Filter)
    {
        throw std::runtime_error{"Asked for output F when not finding gradient w.r.t. F"};
    }
    builder::ContractionOutput ret{F()};
    ret.add_indices({co(), ci()});
    for (const auto& xfi : xfs())
    {
        ret.add_indices({xfi});
    }
    ret.add_dims({CO(), CI()});
    for (const auto& XFi : XFs())
    {
        ret.add_dims({XFi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::ContractionOutput
    ngraph::runtime::plaidml::ConvPoolFormatter::I_out_body()
{
    if (m_deriv != DerivType::Data)
    {
        throw std::runtime_error{"Asked to construct I as output in forward pass"};
    }
    std::string result_name;
    if (m_op == OpType::AvgPool)
    {
        result_name = "DI";
    }
    else
    {
        result_name = I();
    }
    builder::ContractionOutput ret{result_name};
    ret.add_indices({n()});
    if (m_op == OpType::Conv)
    {
        ret.add_indices({ci()});
    }
    else
    {
        ret.add_indices({c()});
    }
    for (const auto& xii : xis())
    {
        ret.add_indices({xii});
    }
    ret.add_dims({N()});
    if (m_op == OpType::Conv)
    {
        ret.add_dims({CI()});
    }
    else
    {
        ret.add_dims({C()});
    }
    for (const auto& XIi : XIs())
    {
        ret.add_dims({XIi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::ContractionOutput
    ngraph::runtime::plaidml::ConvPoolFormatter::O_out_body()
{
    if (m_deriv != DerivType::None && m_op == OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct O as output in gradient pass"};
    }
    std::string name;
    if (m_op == OpType::AvgPool)
    {
        // Special name to allow final division for AvgPool
        name = "S";
    }
    else if (m_op == OpType::MaxPool && m_deriv == DerivType::Data)
    {
        // Special name since forward output is intermediate
        name = "Y";
    }
    else
    {
        name = O();
    }
    builder::ContractionOutput ret{name};
    ret.add_indices({n()});
    if (m_op == OpType::Conv)
    {
        ret.add_indices({co()});
    }
    else
    {
        ret.add_indices({c()});
    }
    for (const auto& xoi : xos())
    {
        ret.add_indices({xoi});
    }
    ret.add_dims({N()});
    if (m_op == OpType::Conv)
    {
        ret.add_dims({CO()});
    }
    else
    {
        ret.add_dims({C()});
    }
    for (const auto& XOi : XOs())
    {
        ret.add_dims({XOi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::ContractionInput
    ngraph::runtime::plaidml::ConvPoolFormatter::F_in_body()
{
    if (m_op != OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct filter F for pooling operation"};
    }
    if (m_deriv == DerivType::Filter)
    {
        throw std::runtime_error{"Asked to construct F as input when computing its gradient"};
    }
    builder::ContractionInput ret{F()};
    ret.add_indices({co(), ci()});
    for (const auto& xfi : xfs())
    {
        ret.add_indices({xfi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::ContractionInput
    ngraph::runtime::plaidml::ConvPoolFormatter::I_in_body()
{
    if (m_deriv == DerivType::Data && m_op == OpType::Conv)
    {
        throw std::runtime_error{"Asked to construct I as input when computing its gradient"};
    }
    builder::ContractionInput ret{"I"};
    ret.add_indices({n()});
    if (m_op == OpType::Conv)
    {
        ret.add_indices({ci()});
    }
    else
    {
        ret.add_indices({c()});
    }
    for (const auto& xii : xis())
    {
        ret.add_indices({xii});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::ContractionInput
    ngraph::runtime::plaidml::ConvPoolFormatter::O_in_body()
{
    if (m_deriv == DerivType::None)
    {
        throw std::runtime_error{"Asked to construct O as input in forward pass"};
    }
    std::string result_name;
    if (m_op == OpType::AvgPool)
    {
        result_name = "S";
    }
    else
    {
        result_name = O();
    }
    builder::ContractionInput ret{result_name};
    ret.add_indices({n()});
    if (m_op == OpType::Conv)
    {
        ret.add_indices({co()});
    }
    else
    {
        ret.add_indices({c()});
    }
    for (const auto& xoi : xos())
    {
        ret.add_indices({xoi});
    }
    return ret;
}

ngraph::runtime::plaidml::builder::UnaryContraction
    ngraph::runtime::plaidml::ConvPoolFormatter::Broadcast_Ones()
{
    if (m_op != OpType::AvgPool)
    {
        throw std::runtime_error{"Broadcast_Ones should only be used for AvgPool"};
    }
    builder::UnaryContraction ret{"="};
    builder::ContractionOutput ones{"Ones"};
    ones.add_indices("o", 0, m_rank);
    for (const auto& XIi : XIs())
    {
        ones.add_dims({XIi});
    }
    ret.set(ones);
    ret.set(builder::ContractionInput{"One"});
    return ret;
}

ngraph::runtime::plaidml::builder::UnaryContraction
    ngraph::runtime::plaidml::ConvPoolFormatter::Count()
{
    if (m_op != OpType::AvgPool)
    {
        throw std::runtime_error{"Count should only be used for AvgPool"};
    }
    builder::UnaryContraction ret{"+"};
    builder::ContractionOutput count{"Count"};
    for (const auto& xoi : xos())
    {
        count.add_indices({xoi});
    }
    for (const auto& XOi : XOs())
    {
        count.add_dims({XOi});
    }
    builder::ContractionInput ones{"Ones"};
    for (const auto& xii : xis())
    {
        ones.add_indices({xii});
    }
    ret.set(count).set(ones).add_constraints(
        [&](std::back_insert_iterator<std::list<std::string>> out) {
            for (std::size_t idx = 0; idx < m_rank; ++idx)
            {
                std::ostringstream s;
                s << "xf" << idx << " < " << m_window_shape[idx];
                out = s.str();
            }
        });
    return ret;
}

ngraph::runtime::plaidml::builder::UnaryContraction
    ngraph::runtime::plaidml::ConvPoolFormatter::PoolContraction()
{
    std::string agg_op;
    switch (m_op)
    {
    case OpType::AvgPool: agg_op = "+"; break;
    case OpType::MaxPool: agg_op = ">"; break;
    default: throw std::runtime_error("Asked for pool contraction for non-pool op");
    }
    return builder::UnaryContraction{agg_op}
        .set((m_op == OpType::AvgPool && m_deriv == DerivType::Data) ? I_out_body() : O_out_body())
        .set((m_op == OpType::AvgPool && m_deriv == DerivType::Data) ? O_in_body() : I_in_body())
        .add_constraints([&](std::back_insert_iterator<std::list<std::string>> out) {
            for (std::size_t idx = 0; idx < m_rank; ++idx)
            {
                std::ostringstream s;
                s << "xf" << idx << " < " << m_window_shape[idx];
                out = s.str();
            }
        });
}

ngraph::runtime::plaidml::builder::TernaryContraction
    ngraph::runtime::plaidml::ConvPoolFormatter::PoolDerivContraction()
{
    builder::ContractionOutput output{"DI"};
    output.add_indices({n(), c()}).add_dims({N(), C()});
    for (const auto& xii : xis())
    {
        output.add_indices({xii});
    }
    for (const auto& XIi : XIs())
    {
        output.add_dims({XIi});
    }
    builder::ContractionInput input{"I"};
    input.add_indices({n(), c()});
    for (const auto& xii : xis())
    {
        input.add_indices({xii});
    }
    builder::ContractionInput forward_output{"Y"};
    forward_output.add_indices({n(), c()});
    for (const auto& xoi : xos())
    {
        forward_output.add_indices({xoi});
    }
    builder::ContractionInput incoming_deriv{"DO"};
    incoming_deriv.add_indices({n(), c()});
    for (const auto& xoi : xos())
    {
        incoming_deriv.add_indices({xoi});
    }
    return builder::TernaryContraction{"+", "?"}
        .set(output)
        .set_first(input)
        .set_second(forward_output)
        .set_third(incoming_deriv);
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::c()
{
    return "c";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::ci()
{
    return "ci";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::co()
{
    return "co";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::n()
{
    return "n";
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::xfs()
{
    if (m_xfs.empty())
    {
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            s << "xf" << i;
            m_xfs.push_back(s.str());
        }
    }
    return m_xfs;
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::xis()
{
    if (m_xis.empty())
    {
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            s << "(";
            s << m_strides[i] << "*xo" << i;
            s << " + ";
            s << m_filter_dilation[i] << "*xf" << i;
            s << " - " << m_pad_below[i];
            s << ")";
            if (m_data_dilation[i] != 1)
            {
                s << " / " << m_data_dilation[i];
            }
            m_xis.push_back(s.str());
        }
    }
    return m_xis;
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::xos()
{
    if (m_xos.empty())
    {
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            s << "xo" << i;
            m_xos.push_back(s.str());
        }
    }
    return m_xos;
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::C()
{
    return "C";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::CI()
{
    return "CI";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::CO()
{
    return "CO";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::N()
{
    return "N";
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::XFs()
{
    if (m_XFs.empty())
    {
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            if (m_deriv == DerivType::Filter)
            {
                s << m_filters_shape[i + 2];
            }
            else
            {
                s << "XF" << i;
            }
            m_XFs.push_back(s.str());
        }
    }
    return m_XFs;
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::XIs()
{
    if (m_XIs.empty())
    {
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            if (m_deriv == DerivType::Data && m_op == OpType::Conv)
            {
                s << m_data_batch_shape[i + 2];
            }
            else
            {
                s << "XI" << i;
            }
            m_XIs.push_back(s.str());
        }
    }
    return m_XIs;
}

std::vector<std::string> ngraph::runtime::plaidml::ConvPoolFormatter::XOs()
{
    if (m_XOs.empty())
    {
        // TODO: Assumes explicit padding...
        for (int i = 0; i < m_rank; ++i)
        {
            std::ostringstream s;
            if (m_deriv == DerivType::None)
            {
                s << "(";
                s << m_data_dilation[i] << " * ";
                s << "(XI" << i << " - 1) + 1 + ";
                s << m_pad_below[i] + m_pad_above[i];
                if (m_window_shape[i] != 0)
                {
                    s << " - " << m_window_shape[i];
                }
                if (m_op == OpType::Conv)
                {
                    s << " - ";
                    s << "(" << m_filter_dilation[i];
                    s << " * (XF" << i << " - 1) + 1)";
                }
                s << " + " << m_strides[i] << ")";
                s << " / " << m_strides[i];
            }
            else
            {
                s << "XO" << i;
            }
            m_XOs.push_back(s.str());
        }
    }
    return m_XOs;
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::F()
{
    if (m_deriv == DerivType::Filter)
    {
        return "DF";
    }
    return "F";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::I()
{
    if (m_deriv == DerivType::Data && m_op == OpType::Conv)
    {
        return "DI";
    }
    return "I";
}

std::string ngraph::runtime::plaidml::ConvPoolFormatter::O()
{
    if (m_deriv != DerivType::None)
    {
        return "DO";
    }
    return "O";
}
