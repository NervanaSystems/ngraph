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

#include "ngraph/specialize_shapes.hpp"

using namespace ngraph;

std::shared_ptr<Function>
    ngraph::specialize_shapes(std::shared_ptr<Function> f,
                              const std::vector<element::Type>& parameter_element_types,
                              const std::vector<PartialShape>& parameter_shapes)
{
    NGRAPH_CHECK(f->get_parameters().size() == parameter_shapes.size());
    NGRAPH_CHECK(f->get_parameters().size() == parameter_element_types.size());

    NodeMap m;

    for (size_t i = 0; i < parameter_shapes.size(); i++)
    {
        NGRAPH_CHECK(
            parameter_shapes[i].refines(f->get_parameters()[i]->get_output_partial_shape(0)));
        NGRAPH_CHECK(f->get_parameters()[i]->get_element_type().is_dynamic() ||
                     parameter_element_types[i] == f->get_parameters()[i]->get_element_type());

        m[f->get_parameters()[i].get()] =
            std::make_shared<op::Parameter>(parameter_element_types[i], parameter_shapes[i]);
    }

    for (auto old_node : f->get_ordered_ops())
    {
        if (old_node->is_parameter())
        {
            continue;
        }

        NodeVector new_args = old_node->get_arguments();

        for (size_t i = 0; i < new_args.size(); i++)
        {
            new_args[i] = m[new_args[i].get()];
        }

        m[old_node.get()] = old_node->copy_with_new_args(new_args);
    }

    ParameterVector new_parameters = f->get_parameters();
    for (size_t i = 0; i < new_parameters.size(); i++)
    {
        new_parameters[i] = std::static_pointer_cast<op::Parameter>(m[new_parameters[i].get()]);
    }

    ResultVector new_results = f->get_results();
    for (size_t i = 0; i < new_results.size(); i++)
    {
        new_results[i] = std::static_pointer_cast<op::Result>(m[new_results[i].get()]);
    }

    return std::make_shared<Function>(new_results, new_parameters);
}
