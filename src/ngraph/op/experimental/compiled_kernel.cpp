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

#include "ngraph/op/experimental/compiled_kernel.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::CompiledKernel::type_info;

shared_ptr<Node>
    ngraph::op::v0::CompiledKernel::clone_with_new_inputs(const OutputVector& new_args) const
{
    return std::make_shared<CompiledKernel>(m_function, new_args);
}

ngraph::op::v0::CompiledKernel::CompiledKernel(const std::shared_ptr<Function>& function,
                                               const OutputVector& args)
    : Op(args)
    , m_function(clone_function(*function))
{
    size_t i = 0;
    for (auto o : function->get_results())
    {
        set_output_type(i, o->get_output_element_type(0), o->get_output_partial_shape(0));
        i++;
    }
}

ngraph::op::v0::CompiledKernel::CompiledKernel(const NodeVector& node_list,
                                               const OutputVector& outputs,
                                               const OutputVector& args)
    : Op(args)
    , m_node_list(node_list)
    , m_outputs(outputs)
{
    constructor_validate_and_infer_types();
    ParameterVector parameters = encapsulate_nodes();
    set_output_size(m_outputs.size());

    shared_ptr<Function> original = make_shared<Function>(outputs, parameters);
    m_function = clone_function(*original);

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto& o = outputs.at(i);

        if (std::find(node_list.begin(), node_list.end(), o.get_node_shared_ptr()) ==
            node_list.end())
        {
            NODE_VALIDATION_CHECK(this, false, "Node for ", o, " isn't in node_list");
        }
        set_output_type(i, o.get_element_type(), o.get_partial_shape());
    }
}

ParameterVector ngraph::op::v0::CompiledKernel::encapsulate_nodes()
{
    std::unordered_set<std::shared_ptr<Node>> node_set(m_node_list.begin(), m_node_list.end());

    // Go through each non-CK user of input to CK
    int ck_arg_idx = 0;
    ParameterVector internal_parameters;
    for (Output<Node> arg_output : input_values())
    {
        auto temp_input_param = std::make_shared<ngraph::op::v0::Parameter>(
            arg_output.get_element_type(), arg_output.get_partial_shape());
        internal_parameters.push_back(temp_input_param);
        for (Input<Node> input : arg_output.get_target_inputs())
        {
            auto user = input.get_node();
            if (!as_type<op::v0::CompiledKernel>(user) &&
                node_set.find(user->shared_from_this()) != node_set.end())
            {
                arg_output.remove_target_input(input);
                // Use a dummy Parameter as input for now, will replace later with the correct
                // one.
                input.replace_source_output(temp_input_param->output(0));
                insert_to_input_map(temp_input_param, ck_arg_idx);
            }
        }
        ck_arg_idx++;
    }
    return internal_parameters;
}

void ngraph::op::v0::CompiledKernel::insert_to_input_map(std::shared_ptr<Node> node,
                                                         size_t ck_arg_idx)
{
    m_input_map.emplace(node, ck_arg_idx);
}

std::shared_ptr<ngraph::Function> ngraph::op::v0::CompiledKernel::get_function()
{
    return m_function;
}
