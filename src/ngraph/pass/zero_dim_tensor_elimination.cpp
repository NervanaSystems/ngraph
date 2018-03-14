/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/sum.hpp"
#include "zero_dim_tensor_elimination.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static std::shared_ptr<Node> build_scalar_constant(std::string val, std::shared_ptr<Node> n)
{
    auto cvals = std::vector<std::string>(shape_size(n->get_shape()), val);
    return std::make_shared<op::Constant>(n->get_element_type(), n->get_shape(), cvals);
}

static std::shared_ptr<Node> return_zero_const(std::shared_ptr<Node> n)
{
    return build_scalar_constant(std::string("0"), n);
}

static std::shared_ptr<Node> return_one_const(std::shared_ptr<Node> n)
{
    return build_scalar_constant(std::string("1"), n);
}

static std::shared_ptr<Node> return_broadcast(std::shared_ptr<Node> n)
{
    AxisSet axes{};
    for (size_t i = 0; i < n->get_shape().size(); i++)
    {
        axes.insert(i);
    }
    return std::make_shared<op::Broadcast>(n->get_input_op(1), n->get_shape(), axes);
}

static const std::unordered_map<std::type_index,
                                std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>>
    ops_to_const_values{
        {TI(ngraph::op::AvgPool), return_zero_const},
        {TI(ngraph::op::MaxPool), return_zero_const},
        {TI(ngraph::op::Convolution), return_zero_const},
        {TI(ngraph::op::Sum), return_zero_const},
        {TI(ngraph::op::Product), return_one_const},
        {TI(ngraph::op::Pad), return_broadcast},
    };

static bool has_zero_dim(std::shared_ptr<Node> node)
{
    if (node->get_output_size() != 1)
    {
        throw ngraph_error("has_zero_dim is called on multi-output op");
    }
    return shape_size(node->get_shape()) == 0;
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

    //all zero-length ops should be in a result set
    //if we remove all such nodes included in the result set
    //from zero_length_nodes and there are still nodes left
    //(in zero_length_nodes), this means we have INTERNAL
    //zero-length nodes (which violates our assumption)
    for (auto r : f->get_results())
    {
        auto n = r->get_input_op(0);
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
    //we need to go over all nodes since we could have sum or any other 0-length-tensor-to scalar op
    //as an internal node (i.e. a node that isn't an argument to `op::Result`)
    for (auto n : f->get_ordered_ops())
    {
        //don't try to replace `op::Result`
        //all multi-output feed into `GetOutputElement`
        //if any `GetOutputElement` is zero-length
        //we replace it w/ a signalling constant
        //so we don't have to deal w/ multi-output nodes directly
        if (n->is_output() || n->is_parameter() || n->get_outputs().size() > 1)
        {
            continue;
        }

        if (has_zero_dim(n))
        {
            //we don't have to create constants every time but this is the easiest
            //and it's CSE's job to eliminate the same ones
            auto cvals = std::vector<std::string>(0);
            auto constant =
                std::make_shared<op::Constant>(n->get_element_type(), n->get_shape(), cvals);
            replace_node(n, constant);
            NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << constant->get_name();
            replaced = true;
            continue;
        }

        const Node& node = *n;
        auto entry = ops_to_const_values.find(TI(node));

        if (entry == ops_to_const_values.end() || !has_zero_dim(n->get_input_op(0)))
        {
            continue;
        }

        auto new_node = entry->second(n);
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
