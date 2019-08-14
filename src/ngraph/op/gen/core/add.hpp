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
                class Add;
            } // namespace core
        }     // namespace gen
    }         // namespace op
} // namespace ngraph

// Elementwise addition operation
class ::ngraph::op::gen::core::Add final : public ::ngraph::op::util::GenOp
{
public:
    NGRAPH_API static const ::std::string type_name;
    const std::string& description() const final override { return type_name; }
    class Builder;
    static ::std::shared_ptr<::ngraph::Node>
        build(const ::ngraph::OutputVector& source_outputs,
              const ::std::vector<const ::ngraph::AttributeBase*>& attributes);
    Add() = default;
    Add(const ::ngraph::Output<::ngraph::Node>& x,
        const ::ngraph::Output<::ngraph::Node>& y,
        const AutoBroadcastSpec& autobroadcast)
        : ::ngraph::op::util::GenOp(::ngraph::OutputVector{x, y})
        , m_autobroadcast(autobroadcast)
    {
    }
    Add(const ::ngraph::OutputVector& source_outputs,
        const ::std::vector<const ::ngraph::AttributeBase*>& attributes)
        : ::ngraph::op::util::GenOp(source_outputs)
    {
        NGRAPH_CHECK(source_outputs.size() == 2,
                     "Source output count should be 2, not ",
                     source_outputs.size());
        NGRAPH_CHECK(
            attributes.size() == 1, "Attribute count should be 1, not ", attributes.size());
        NGRAPH_CHECK(
            attributes[0]->has_type<AutoBroadcastSpec>(),
            "Attribute 0 (name: autobroadcast) has incorrect type (AutoBroadcastSpec expected)");
        m_autobroadcast.set(attributes[0]->as_type<AutoBroadcastSpec>().get());
    }
    ::ngraph::Input<::ngraph::Node> get_x() { return input(0); }
    ::ngraph::Input<::ngraph::Node> get_y() { return input(1); }
    ::ngraph::Input<const ::ngraph::Node> get_x() const { return input(0); }
    ::ngraph::Input<const ::ngraph::Node> get_y() const { return input(1); }
    ::ngraph::Output<::ngraph::Node> get_z() { return output(0); }
    ::ngraph::Output<const ::ngraph::Node> get_z() const { return output(0); }
    const AutoBroadcastSpec& get_autobroadcast() const { return m_autobroadcast.get(); }
    void set_autobroadcast(const AutoBroadcastSpec& autobroadcast)
    {
        m_autobroadcast.set(autobroadcast);
    }
    ::std::vector<::std::string> get_argument_keys() const final override
    {
        return ::std::vector<::std::string>{"x", "y"};
    }
    ::std::vector<::std::string> get_result_keys() const final override
    {
        return ::std::vector<::std::string>{"z"};
    }
    ::std::vector<::std::string> get_attribute_keys() const final override
    {
        return ::std::vector<::std::string>{"autobroadcast"};
    }
    ::ngraph::Input<const ::ngraph::Node>
        get_argument(const ::std::string& name) const final override
    {
        if (name == "x")
        {
            return input(0);
        }
        else if (name == "y")
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
        if (name == "x")
        {
            return input(0);
        }
        else if (name == "y")
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
        if (name == "z")
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
        if (name == "z")
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
        if (name == "autobroadcast")
        {
            return m_autobroadcast;
        }
        else
        {
            NGRAPH_CHECK(false, "get_attribute: Invalid attribute name ", name);
        }
    }
    bool is_commutative() const final override { return true; }
    bool has_state() const final override { return false; }
    ::std::shared_ptr<::ngraph::Node>
        copy_with_new_args(const ::ngraph::NodeVector& inputs) const final override
    {
        NGRAPH_CHECK(inputs.size() == 2, "New argument count should be 2, not ", inputs.size());
        ::std::shared_ptr<::ngraph::Node> new_node =
            ::std::make_shared<Add>(inputs[0], inputs[1], m_autobroadcast.get());
        // TODO: control deps
        return new_node;
    }

private:
    ::ngraph::Attribute<AutoBroadcastSpec> m_autobroadcast;
    NGRAPH_API static bool s_registered;
};
class ::ngraph::op::gen::core::Add::Builder final : public ::ngraph::GenOpBuilder
{
public:
    ::std::shared_ptr<::ngraph::Node>
        build(const ::ngraph::OutputVector& source_outputs,
              const ::std::vector<const ::ngraph::AttributeBase*>& attributes) const final override
    {
        return ::ngraph::op::gen::core::Add::build(source_outputs, attributes);
    }
};
