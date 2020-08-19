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

#include "ngraph/specialize_function.hpp"
#include <pass/constant_folding.hpp>
#include "ngraph/op/constant.hpp"
#include "ngraph/op/tensor_iterator.hpp"

using namespace ngraph;

std::shared_ptr<Function>
    ngraph::specialize_function(std::shared_ptr<Function> f,
                                const std::vector<element::Type>& parameter_element_types,
                                const std::vector<PartialShape>& parameter_shapes,
                                const std::vector<void*>& parameter_values)
{
    return specialize_function(
        f, parameter_element_types, parameter_shapes, parameter_values, false, false);
}

std::shared_ptr<Function>
    ngraph::specialize_function(std::shared_ptr<Function> f,
                                const std::vector<element::Type>& parameter_element_types,
                                const std::vector<PartialShape>& parameter_shapes,
                                const std::vector<void*>& parameter_values,
                                bool constant_folding,
                                bool share_constants)
{
    NGRAPH_CHECK(f->get_parameters().size() == parameter_shapes.size());
    NGRAPH_CHECK(f->get_parameters().size() == parameter_element_types.size());
    NGRAPH_CHECK(f->get_parameters().size() == parameter_values.size());

    NodeMap m;

    for (size_t i = 0; i < parameter_shapes.size(); i++)
    {
        NGRAPH_CHECK(f->get_parameters()[i]->get_element_type().is_dynamic() ||
                     parameter_element_types[i] == f->get_parameters()[i]->get_element_type());

        if (parameter_values[i] != nullptr && parameter_shapes[i].is_static() &&
            parameter_element_types[i].is_static())
        {
            m[f->get_parameters()[i].get()] = std::make_shared<op::v0::Constant>(
                parameter_element_types[i], parameter_shapes[i].to_shape(), parameter_values[i]);
        }
        else
        {
            m[f->get_parameters()[i].get()] = std::make_shared<op::v0::Parameter>(
                parameter_element_types[i], parameter_shapes[i]);
        }
        m[f->get_parameters()[i].get()]->get_rt_info() = f->get_parameters()[i]->get_rt_info();
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

        if (share_constants && as_type_ptr<op::v0::Constant>(old_node))
        {
            m[old_node.get()] = old_node;
        }
        else
        {
            m[old_node.get()] = old_node->copy_with_new_inputs(new_args);
            //  TODO: workaround for shape inference, delete it after fix
            if (::ngraph::as_type_ptr<ngraph::op::v0::TensorIterator>(m[old_node.get()]))
            {
                m[old_node.get()]->validate_and_infer_types();
            }
            m[old_node.get()]->get_rt_info() = old_node->get_rt_info();
        }

        m[old_node.get()]->set_friendly_name(old_node->get_friendly_name());
    }

    ParameterVector new_parameters = f->get_parameters();
    for (size_t i = 0; i < new_parameters.size(); i++)
    {
        auto name = new_parameters[i]->get_friendly_name();
        new_parameters[i] = as_type_ptr<op::v0::Parameter>(m[new_parameters[i].get()]);

        // If the replacement for a Parameter is not itself a Parameter, we must have replaced it
        // with a constant. We will insert a dead Parameter into the clone's parameters, in order
        // to maintain the arity of the original function.
        if (new_parameters[i] == nullptr)
        {
            new_parameters[i] = std::make_shared<op::v0::Parameter>(parameter_element_types[i],
                                                                    parameter_shapes[i]);
        }
        new_parameters[i]->set_friendly_name(name);
    }

    ResultVector new_results = f->get_results();
    for (size_t i = 0; i < new_results.size(); i++)
    {
        auto name = new_results[i]->get_friendly_name();
        new_results[i] = std::static_pointer_cast<op::v0::Result>(m[new_results[i].get()]);
        new_results[i]->set_friendly_name(name);
    }

    auto function = std::make_shared<Function>(new_results, new_parameters);
    if (constant_folding)
    {
        ngraph::pass::ConstantFolding().run_on_function(function);
    }
    return function;
}
