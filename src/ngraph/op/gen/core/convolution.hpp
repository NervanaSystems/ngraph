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
class ::ngraph::op::gen::core::Convolution final : public ::ngraph::op::util::GenOp
{
public:
    NGRAPH_API static const ::std::string type_name;
    const std::string& description() const final override { return type_name; }
    class Builder;
    static ::std::shared_ptr<::ngraph::Node>
        build(const ::ngraph::OutputVector& source_outputs,
              const ::std::vector<const ::ngraph::AttributeBase*>& attributes);
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
    Convolution(const ::ngraph::OutputVector& source_outputs,
                const ::std::vector<const ::ngraph::AttributeBase*>& attributes)
        : ::ngraph::op::util::GenOp(source_outputs)
    {
        NGRAPH_CHECK(source_outputs.size() == 2,
                     "Source output count should be 2, not ",
                     source_outputs.size());
        NGRAPH_CHECK(
            attributes.size() == 6, "Attribute count should be 6, not ", attributes.size());
        NGRAPH_CHECK(attributes[0]->has_type<Strides>(),
                     "Attribute 0 (name: strides) has incorrect type (Strides expected)");
        NGRAPH_CHECK(attributes[1]->has_type<Strides>(),
                     "Attribute 1 (name: dilation) has incorrect type (Strides expected)");
        NGRAPH_CHECK(attributes[2]->has_type<Strides>(),
                     "Attribute 2 (name: data_dilation) has incorrect type (Strides expected)");
        NGRAPH_CHECK(
            attributes[3]->has_type<CoordinateDiff>(),
            "Attribute 3 (name: padding_before) has incorrect type (CoordinateDiff expected)");
        NGRAPH_CHECK(
            attributes[4]->has_type<CoordinateDiff>(),
            "Attribute 4 (name: padding_after) has incorrect type (CoordinateDiff expected)");
        NGRAPH_CHECK(attributes[5]->has_type<PadType>(),
                     "Attribute 5 (name: pad_type) has incorrect type (PadType expected)");
        m_strides.set(attributes[0]->as_type<Strides>().get());
        m_dilation.set(attributes[1]->as_type<Strides>().get());
        m_data_dilation.set(attributes[2]->as_type<Strides>().get());
        m_padding_before.set(attributes[3]->as_type<CoordinateDiff>().get());
        m_padding_after.set(attributes[4]->as_type<CoordinateDiff>().get());
        m_pad_type.set(attributes[5]->as_type<PadType>().get());
    }
    ::ngraph::Input<::ngraph::Node> get_data() { return input(0); }
    ::ngraph::Input<::ngraph::Node> get_filter() { return input(1); }
    ::ngraph::Input<const ::ngraph::Node> get_data() const { return input(0); }
    ::ngraph::Input<const ::ngraph::Node> get_filter() const { return input(1); }
    ::ngraph::Output<::ngraph::Node> get_output() { return output(0); }
    ::ngraph::Output<const ::ngraph::Node> get_output() const { return output(0); }
    const Strides& get_strides() const { return m_strides.get(); }
    const Strides& get_dilation() const { return m_dilation.get(); }
    const Strides& get_data_dilation() const { return m_data_dilation.get(); }
    const CoordinateDiff& get_padding_before() const { return m_padding_before.get(); }
    const CoordinateDiff& get_padding_after() const { return m_padding_after.get(); }
    const PadType& get_pad_type() const { return m_pad_type.get(); }
    void set_strides(const Strides& strides) { m_strides.set(strides); }
    void set_dilation(const Strides& dilation) { m_dilation.set(dilation); }
    void set_data_dilation(const Strides& data_dilation) { m_data_dilation.set(data_dilation); }
    void set_padding_before(const CoordinateDiff& padding_before)
    {
        m_padding_before.set(padding_before);
    }
    void set_padding_after(const CoordinateDiff& padding_after)
    {
        m_padding_after.set(padding_after);
    }
    void set_pad_type(const PadType& pad_type) { m_pad_type.set(pad_type); }
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
    ::ngraph::Input<const ::ngraph::Node>
        get_argument(const ::std::string& name) const final override
    {
        if (name == "data")
        {
            return input(0);
        }
        else if (name == "filter")
        {
            return input(1);
        }
        else
        {
            NGRAPH_CHECK(false, "get_argument: Invalid argument name ", name);
        }
    }
    ::ngraph::Input<::ngraph::Node> get_argument(const ::std::string& name) final override
    {
        if (name == "data")
        {
            return input(0);
        }
        else if (name == "filter")
        {
            return input(1);
        }
        else
        {
            NGRAPH_CHECK(false, "get_argument: Invalid argument name ", name);
        }
    }
    ::ngraph::Output<const ::ngraph::Node>
        get_result(const ::std::string& name) const final override
    {
        if (name == "output")
        {
            return output(0);
        }
        else
        {
            NGRAPH_CHECK(false, "get_result: Invalid result name ", name);
        }
    }
    ::ngraph::Output<::ngraph::Node> get_result(const ::std::string& name) final override
    {
        if (name == "output")
        {
            return output(0);
        }
        else
        {
            NGRAPH_CHECK(false, "get_result: Invalid result name ", name);
        }
    }
    const ::ngraph::AttributeBase& get_attribute(const ::std::string& name) const final override
    {
        if (name == "strides")
        {
            return m_strides;
        }
        else if (name == "dilation")
        {
            return m_dilation;
        }
        else if (name == "data_dilation")
        {
            return m_data_dilation;
        }
        else if (name == "padding_before")
        {
            return m_padding_before;
        }
        else if (name == "padding_after")
        {
            return m_padding_after;
        }
        else if (name == "pad_type")
        {
            return m_pad_type;
        }
        else
        {
            NGRAPH_CHECK(false, "get_attribute: Invalid attribute name ", name);
        }
    }
    bool is_commutative() const final override { return false; }
    bool has_state() const final override { return false; }
    ::std::shared_ptr<::ngraph::Node>
        copy_with_new_args(const ::ngraph::NodeVector& inputs) const final override
    {
        NGRAPH_CHECK(inputs.size() == 2, "New argument count should be 2, not ", inputs.size());
        ::std::shared_ptr<::ngraph::Node> new_node =
            ::std::make_shared<Convolution>(inputs[0],
                                            inputs[1],
                                            m_strides.get(),
                                            m_dilation.get(),
                                            m_data_dilation.get(),
                                            m_padding_before.get(),
                                            m_padding_after.get(),
                                            m_pad_type.get());
        // TODO: control deps
        return new_node;
    }

private:
    ::ngraph::Attribute<Strides> m_strides;
    ::ngraph::Attribute<Strides> m_dilation;
    ::ngraph::Attribute<Strides> m_data_dilation;
    ::ngraph::Attribute<CoordinateDiff> m_padding_before;
    ::ngraph::Attribute<CoordinateDiff> m_padding_after;
    ::ngraph::Attribute<PadType> m_pad_type;
    NGRAPH_API static bool s_registered;
};
class ::ngraph::op::gen::core::Convolution::Builder final : public ::ngraph::GenOpBuilder
{
public:
    ::std::shared_ptr<::ngraph::Node>
        build(const ::ngraph::OutputVector& source_outputs,
              const ::std::vector<const ::ngraph::AttributeBase*>& attributes) const final override
    {
        return ::ngraph::op::gen::core::Convolution::build(source_outputs, attributes);
    }
};
