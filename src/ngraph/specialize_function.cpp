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

#include "ngraph/specialize_function.hpp"
#include "ngraph/op/constant.hpp"

using namespace ngraph;

std::shared_ptr<Function>
    ngraph::specialize_function(std::shared_ptr<Function> f,
                                const std::vector<element::Type>& parameter_element_types,
                                const std::vector<PartialShape>& parameter_shapes,
                                const std::vector<void*>& parameter_values)
{
    NGRAPH_CHECK(f->get_parameters().size() == parameter_shapes.size());
    NGRAPH_CHECK(f->get_parameters().size() == parameter_element_types.size());
    NGRAPH_CHECK(f->get_parameters().size() == parameter_values.size());

    NodeMap m;

    for (size_t i = 0; i < parameter_shapes.size(); i++)
    {
        NGRAPH_CHECK(
            parameter_shapes[i].refines(f->get_parameters()[i]->get_output_partial_shape(0)));
        NGRAPH_CHECK(f->get_parameters()[i]->get_element_type().is_dynamic() ||
                     parameter_element_types[i] == f->get_parameters()[i]->get_element_type());

        if (parameter_values[i] != nullptr && parameter_shapes[i].is_static() &&
            parameter_element_types[i].is_static())
        {
            m[f->get_parameters()[i].get()] = std::make_shared<op::Constant>(
                parameter_element_types[i], parameter_shapes[i].to_shape(), parameter_values[i]);
        }
        else
        {
            m[f->get_parameters()[i].get()] =
                std::make_shared<op::Parameter>(parameter_element_types[i], parameter_shapes[i]);
        }
    }

    for (auto old_node : f->get_ordered_ops())
    {
        if (old_node->is_parameter())
        {
            continue;
        }

        OutputVector new_args;
        for (auto input : old_node->inputs())
        {
            auto output = input.get_source_output();
            new_args.push_back(output.for_node(m[output.get_node()]));
        }
        m[old_node.get()] = old_node->copy_with_new_inputs(new_args);
    }

    ParameterVector new_parameters = f->get_parameters();
    for (size_t i = 0; i < new_parameters.size(); i++)
    {
        new_parameters[i] = as_type_ptr<op::Parameter>(m[new_parameters[i].get()]);

        // If the replacement for a Parameter is not itself a Parameter, we must have replaced it
        // with a constant. We will insert a dead Parameter into the clone's parameters, in order
        // to maintain the arity of the original function.
        if (new_parameters[i] == nullptr)
        {
            new_parameters[i] =
                std::make_shared<op::Parameter>(parameter_element_types[i], parameter_shapes[i]);
        }
    }

    ResultVector new_results = f->get_results();
    for (size_t i = 0; i < new_results.size(); i++)
    {
        new_results[i] = std::static_pointer_cast<op::Result>(m[new_results[i].get()]);
    }

    return std::make_shared<Function>(new_results, new_parameters);
}
