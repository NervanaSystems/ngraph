//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"
#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/util.hpp"
#include "node_factory.hpp"

namespace
{
    class DictAttributeDeserializer : public ngraph::AttributeVisitor
    {
    public:
        DictAttributeDeserializer(const py::dict& attributes)
            : m_attributes(attributes)
        {
        }

        virtual void on_attribute(const std::string& name, std::string& value) override
        {
            if (m_attributes.contains(name))
            {
                value = m_attributes[name.c_str()].cast<std::string>();
            }
        }
        virtual void on_attribute(const std::string& name, bool& value) override
        {
            if (m_attributes.contains(name))
            {
                value = m_attributes[name.c_str()].cast<bool>();
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<void>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                NGRAPH_CHECK(
                    false, "No AttributeVisitor support for accessing attribute named: ", name);
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::string>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<std::string>());
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::int64_t>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<std::int64_t>());
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<double>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<double>());
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<std::vector<std::int64_t>>());
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<std::string>>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<std::vector<std::string>>());
            }
        }
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<float>>& adapter) override
        {
            if (m_attributes.contains(name))
            {
                adapter.set(m_attributes[name.c_str()].cast<std::vector<float>>());
            }
        }

    protected:
        const py::dict& m_attributes;
    };

    class NodeFactory
    {
    public:
        NodeFactory() {}
        NodeFactory(const std::string& opset_name)
            : m_opset{get_opset(opset_name)}
        {
        }

        std::shared_ptr<ngraph::Node> create(const std::string op_type_name,
                                             const ngraph::NodeVector& arguments,
                                             const py::dict& attributes = py::dict())
        {
            std::shared_ptr<ngraph::Node> op_node =
                std::shared_ptr<ngraph::Node>(m_opset.create(op_type_name));

            NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
            DictAttributeDeserializer visitor(attributes);

            op_node->set_arguments(arguments);
            if (op_node->is_constant())
            {
                std::vector<size_t> shape = attributes["shape"].cast<std::vector<size_t>>();
                ngraph::element::Type el_type = ngraph::as_enum<ngraph::element::Type_t>(
                    attributes["element_type"].cast<std::string>());
                const void* data = attributes["value"].cast<void*>();
                // TODO: How to select correct opset version?
                op_node = std::make_shared<ngraph::op::v0::Constant>(el_type, shape, data);
            }
            else
            {
                op_node->visit_attributes(visitor);
            }
            op_node->constructor_validate_and_infer_types();

            return op_node;
        }

    private:
        const ngraph::OpSet& get_opset(const std::string& opset_name)
        {
            std::string opset_name_ = opset_name;
            std::transform(opset_name_.begin(), opset_name_.end(), opset_name_.begin(), ::tolower);

            if (opset_name_ == "opset0")
            {
                return ngraph::get_opset0();
            }
            else if (opset_name_ == "opset1")
            {
                return ngraph::get_opset1();
            }
            else if (opset_name_ == "opset2")
            {
                return ngraph::get_opset2();
            }
            else
            {
                throw ngraph::ngraph_error("Unsupported opset version requested.");
            }
        }

        const ngraph::OpSet& m_opset{ngraph::get_opset0()};
    };
}

namespace py = pybind11;

void regclass_pyngraph_NodeFactory(py::module m)
{
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", &NodeFactory::create);

    node_factory.def("__repr__", [](const NodeFactory& self) { return "<NodeFactory>"; });
}
