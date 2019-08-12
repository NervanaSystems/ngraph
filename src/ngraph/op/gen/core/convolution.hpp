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

#pragma once

#include "ngraph/op/util/gen_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace gen
        {
            namespace core
            {
                class Convolution;
            } // namespace core
        }     // namespace gen
    }         // namespace op
} // namespace ngraph

// Batched convolution operation
class ::ngraph::op::gen::core::Convolution : public ::ngraph::op::util::GenOp
{
public:
    NGRAPH_API static const ::std::string type_name;
    const std::string& description() const final override { return type_name; }
    Convolution() = default;
    Convolution(const ::ngraph::Output<::ngraph::Node>& data,
                const ::ngraph::Output<::ngraph::Node>& filter,
                const Strides& strides,
                const Strides& dilation,
                const Strides& data_dilation,
                const CoordinateDiff& padding_before,
                const CoordinateDiff& padding_after,
                const PadType& pad_type)
        : ::ngraph::op::util::GenOp(::ngraph::OutputVector{data, filter})
        , m_strides(strides)
        , m_dilation(dilation)
        , m_data_dilation(data_dilation)
        , m_padding_before(padding_before)
        , m_padding_after(padding_after)
        , m_pad_type(pad_type)
    {
    }
    ::ngraph::Input<::ngraph::Node> get_data() { return input(0); }
    ::ngraph::Input<::ngraph::Node> get_filter() { return input(1); }
    ::ngraph::Input<const ::ngraph::Node> get_data() const { return input(0); }
    ::ngraph::Input<const ::ngraph::Node> get_filter() const { return input(1); }
    ::ngraph::Output<::ngraph::Node> get_output() { return output(0); }
    ::ngraph::Output<const ::ngraph::Node> get_output() const { return output(0); }
    const Strides& get_strides() const { return m_strides; }
    const Strides& get_dilation() const { return m_dilation; }
    const Strides& get_data_dilation() const { return m_data_dilation; }
    const CoordinateDiff& get_padding_before() const { return m_padding_before; }
    const CoordinateDiff& get_padding_after() const { return m_padding_after; }
    const PadType& get_pad_type() const { return m_pad_type; }
    void set_strides(const Strides& strides) { m_strides = strides; }
    void set_dilation(const Strides& dilation) { m_dilation = dilation; }
    void set_data_dilation(const Strides& data_dilation) { m_data_dilation = data_dilation; }
    void set_padding_before(const CoordinateDiff& padding_before)
    {
        m_padding_before = padding_before;
    }
    void set_padding_after(const CoordinateDiff& padding_after) { m_padding_after = padding_after; }
    void set_pad_type(const PadType& pad_type) { m_pad_type = pad_type; }
    ::std::vector<::std::string> get_argument_keys() const final override
    {
        return ::std::vector<::std::string>{"data", "filter"};
    }
    ::std::vector<::std::string> get_result_keys() const final override
    {
        return ::std::vector<::std::string>{"output"};
    }
    ::std::vector<::std::string> get_attribute_keys() const final override
    {
        return ::std::vector<::std::string>{
            "strides", "dilation", "data_dilation", "padding_before", "padding_after", "pad_type"};
    }
    bool is_commutative() const final override { return false; }
    bool has_state() const final override { return false; }
    ::std::shared_ptr<::ngraph::Node>
        copy_with_new_args(const ::ngraph::NodeVector& inputs) const final override
    {
        // TODO: check input count
        ::std::shared_ptr<::ngraph::Node> new_node =
            ::std::make_shared<Convolution>(inputs[0],
                                            inputs[1],
                                            m_strides,
                                            m_dilation,
                                            m_data_dilation,
                                            m_padding_before,
                                            m_padding_after,
                                            m_pad_type);
        // TODO: control deps
        return new_node;
    }

private:
    Strides m_strides;
    Strides m_dilation;
    Strides m_data_dilation;
    CoordinateDiff m_padding_before;
    CoordinateDiff m_padding_after;
    PadType m_pad_type;
};
