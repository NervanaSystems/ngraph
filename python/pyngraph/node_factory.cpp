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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "node_factory.hpp"
#include "ngraph/node.hpp"
#include "ngraph/output_vector.hpp"

class NodeFactory
{
public:
    NodeFactory()
    {
        std::cout << "Constructor called" << std::endl;
        std::cout << "default opset will be used" << std::endl;
    }

    NodeFactory(std::string opset_name)
    {
        std::cout << "Constructor called" << std::endl;
        std::cout << "opset_name:" << opset_name << std::endl;
    }

    void create(const std::string op_type_name, ngraph::NodeVector arguments, py::dict attributes)
    {
        std::cout << "create called" << std::endl;
        std::cout << "op_type_name: " << op_type_name << std::endl;


        for (auto item : arguments)
        {
            std::cout << "argument: " << item << std::endl;
        }

        std::cout << "attributes: " << attributes << std::endl;

        for (auto item : attributes)
        {
            std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
            std::cout << "key: " << item.first << ", class=" << typeid(item.second).name() << std::endl;
        }
    }
};


namespace py = pybind11;

void regclass_pyngraph_NodeFactory(py::module m)
{
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", &NodeFactory::create);

    node_factory.def("__repr__", [](const NodeFactory& self) {
        return "<NodeFactory>";
    });
}
