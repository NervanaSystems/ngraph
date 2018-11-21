//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory>
#include <set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"
#include "zero_dim_tensor_elimination.hpp"

using namespace ngraph;

static bool has_zero_dim(std::shared_ptr<Node> node)
{
    if (node->get_output_size() != 1)
    {
        throw ngraph_error("has_zero_dim is called on multi-output op");
    }

    const auto& shape = node->get_shape();
    return std::find(shape.begin(), shape.end(), 0) != shape.end();
}

static bool verify_no_internal_zero_length_ops(std::shared_ptr<ngraph::Function> f)
{
    std::set<std::shared_ptr<Node>> zero_length_nodes;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter() || n->get_outputs().size() > 1)
        {
            continue;
        }

        if (has_zero_dim(n))
        {
            zero_length_nodes.insert(n);
        }
    }

    // all zero-length ops should be in a result set
    // if we remove all such nodes included in the result set
    // from zero_length_nodes and there are still nodes left
    //(in zero_length_nodes), this means we have INTERNAL
    // zero-length nodes (which violates our assumption)
    for (auto r : f->get_results())
    {
        auto n = r->get_argument(0);
        if (zero_length_nodes.count(n) != 0)
        {
            zero_length_nodes.erase(n);
        }
    }

    return zero_length_nodes.size() > 0;
}

bool ngraph::pass::ZeroDimTensorElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    auto cvals = std::vector<std::string>(0);
    // we need to go over all nodes since we could have sum or any other 0-length-tensor-to scalar op
    // as an internal node (i.e. a node that isn't an argument to `op::Result`)
    for (auto n : f->get_ordered_ops())
    {
        // don't try to replace `op::Result`
        // all multi-output feed into `GetOutputElement`
        // if any `GetOutputElement` is zero-length
        // we replace it w/ a signalling constant
        // so we don't have to deal w/ multi-output nodes directly
        if (n->is_output() || n->is_parameter() || n->get_outputs().size() > 1)
        {
            continue;
        }

        if (has_zero_dim(n))
        {
            // we don't have to create constants every time but this is the easiest
            // and it's CSE's job to eliminate the same ones
            auto constant =
                std::make_shared<op::Constant>(n->get_element_type(), n->get_shape(), cvals);
            replace_node(n, constant);
            NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << constant->get_name();
            replaced = true;
            continue;
        }

        if (n->get_inputs().size() == 0)
        {
            continue;
        }

        if (auto concat = std::dynamic_pointer_cast<op::Concat>(n))
        {
            NodeVector non_zero_dim_args;
            for (auto arg : concat->get_arguments())
            {
                if (!has_zero_dim(arg))
                {
                    non_zero_dim_args.push_back(arg);
                }
            }

            if (non_zero_dim_args.size() < concat->get_inputs().size())
            {
                auto new_concat = concat->copy_with_new_args(non_zero_dim_args);
                NGRAPH_DEBUG << " Replacing " << n->get_name() << " with "
                             << new_concat->get_name();
                ngraph::replace_node(concat, new_concat);
                continue;
            }
        }

        auto arg = n->get_inputs().at(0).get_output().get_node();

        if (arg->get_outputs().size() != 1 || !has_zero_dim(arg))
        {
            continue;
        }

        auto new_node = n->get_default_value();

        if (!new_node)
        {
            continue;
        }

        replaced = true;
        NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << new_node->get_name();
        replace_node(n, new_node);
    }

    if (verify_no_internal_zero_length_ops(f))
    {
        throw ngraph_error("there were internal zero-length nodes in a graph");
    }

    return replaced;
}
