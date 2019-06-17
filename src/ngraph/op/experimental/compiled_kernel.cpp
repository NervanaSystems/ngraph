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

#include "ngraph/op/experimental/compiled_kernel.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> ngraph::op::CompiledKernel::copy_with_new_args(const NodeVector& new_args) const
{
    auto args = get_arguments();
    if (new_args.size() != args.size())
    {
        throw ngraph_error("number of arguments don't match");
    }

    // map inputs
    NodeMap nm;
    for (size_t i = 0; i < args.size(); i++)
    {
        nm[args.at(i).get()] = new_args.at(i);
    }

    NodeVector new_node_list;
    for (auto n : m_node_list)
    {
        NodeVector cur_args;
        for (auto a : n->get_arguments())
        {
            cur_args.push_back(nm.at(a.get()));
        }
        auto new_n = n->copy_with_new_args(cur_args);
        nm[n.get()] = new_n;
        new_node_list.push_back(new_n);
    }

    NodeVector new_outputs;
    for (auto o : m_output_nodes)
    {
        new_outputs.push_back(nm.at(o.get()));
    }

    return std::make_shared<CompiledKernel>(new_node_list, new_outputs, new_args);
}

ngraph::op::CompiledKernel::CompiledKernel(const NodeVector& node_list,
                                           const NodeVector& outputs,
                                           const NodeVector& args)
    : Op("CompiledKernel", check_single_output_args({args}))
    , m_node_list(node_list)
    , m_output_nodes(outputs)
{

    constructor_validate_and_infer_types();

    set_output_size(args.size());
    std::vector<Node*> raw_node_list;
    for (auto node : node_list)
    {
        raw_node_list.push_back(node.get());
    }
    // Replace input edges to sub-graph with output of CK instead. 
    // This ensures the sub-graph is unreachable from the rest of the graph
    unsigned i = 0;
    for (auto arg : args)
    {
        // CK output is identical to corresponding input to sub-graph
        set_output_type(i, arg->get_element_type(), arg->get_shape());
        
        // Find edges from input nodes that go into the sub-graph. Replace them with CK output.
        for (auto output : arg->outputs())
        {
            // make a copy since modifying the inputs list will corrupt the container iterator
            auto inputs = output.get_target_inputs();
            // all inputs that this output feeds
            for (auto use : inputs)
            {
                if (std::find(raw_node_list.begin(), raw_node_list.end(), use.get_node()) != raw_node_list.end())
                {
                    // find uses inside the sub-graph. Replace source with corresponding output of CK
                    use.replace_source_output(this->output(i));
                }
            }
        }
        i++;
    }

}
