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

#include "reshape_sinking.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_set>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

extern template ngraph::AxisVector
    ngraph::apply_permutation<ngraph::AxisVector>(ngraph::AxisVector input,
                                                  ngraph::AxisVector order);

extern template ngraph::Shape ngraph::apply_permutation<ngraph::Shape>(ngraph::Shape input,
                                                                       ngraph::AxisVector order);

using ReshapeMap = std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<op::Reshape>>;

static std::shared_ptr<op::Reshape> combine_reshapes(std::shared_ptr<op::Reshape> r1,
                                                     std::shared_ptr<op::Reshape> r2)
{
    auto default_order = ngraph::get_default_order(r1->get_shape());
    auto perm_r1 = apply_permutation(default_order, r1->get_input_order());
    auto perm_r2 = apply_permutation(perm_r1, r2->get_input_order());
    auto rreshape = std::make_shared<op::Reshape>(r2->get_argument(0), perm_r2, r2->get_shape());
    return rreshape;
}

static void
    insert_reshape(std::shared_ptr<Node> target, std::shared_ptr<Node> reshape, size_t input_index)
{
    auto arg = target->get_inputs().at(input_index).get_output().get_node();
    auto new_reshape = reshape->copy_with_new_args({arg});
    target->get_inputs().at(input_index).replace_output(new_reshape->get_outputs().at(0));
}

std::string describe_reshape(std::shared_ptr<Node> node)
{
    std::stringstream ss;
    auto reshape = std::dynamic_pointer_cast<op::Reshape>(node);
    ss << reshape->get_name()
       << " ( axis order = " << ngraph::vector_to_string(reshape->get_input_order())
       << " , shape = " << vector_to_string(reshape->get_shape()) << " ) "
       << " , child = " << reshape->get_argument(0)->get_name();

    return ss.str();
}

static void delete_reshape(std::shared_ptr<Node> reshape)
{
    NGRAPH_DEBUG << "Removing reshape " << reshape->get_name();
    if (!reshape->get_users().empty())
    {
        ngraph::replace_node(reshape, reshape->get_argument(0));
    }
}

static void mark_reshape_for_deletion(std::shared_ptr<Node> reshape,
                                      std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    NGRAPH_DEBUG << "Marking reshape " << reshape->get_name() << " for deletion";
    reshapes_to_delete.insert(reshape);
}

static std::shared_ptr<op::Reshape> create_default_reshape(std::shared_ptr<Node> n)
{
    auto default_order = ngraph::get_default_order(n->get_shape());
    auto default_reshape = std::make_shared<op::Reshape>(n, default_order, n->get_shape());
    return default_reshape;
}

//compute an axis order that converts the given axis order to default
static AxisSet get_quantization_axes_in_default_order(std::shared_ptr<op::Reshape> arg_reshape,
                                                      const AxisSet& old_axis_set)
{
    auto perm_to_def = ngraph::get_permutation_to_default_order(arg_reshape->get_input_order());
    AxisSet axis_set;
    for (auto axis : old_axis_set)
    {
        axis_set.insert(perm_to_def.at(axis));
    }
    return axis_set;
}

struct Swimmer
{
    descriptor::Input* input;
    std::shared_ptr<op::Reshape> reshape;
};

//Swim is used to push/"swim" reshapes towards paramaters.
//This is typically done for binary ops when
//one operand is in nchw, while  the other one is nhwc
//we prefer nchw since a lot of ngraph ops require this format,
//so keeping things in nchw allows us to eliminate as many reshapes
//as possible
void swim(descriptor::Input* input, std::shared_ptr<op::Reshape> reshape)
{
    Swimmer sw{input, reshape};
    std::list<Swimmer> work_queue;
    work_queue.push_back(sw);

    //TODO: if we support more ops (especially, with >1 args)
    //we will need to keep track of nodes we visited and their reshapes
    while (work_queue.size() > 0)
    {
        auto csw = work_queue.front();
        work_queue.pop_front();
        auto n = csw.input->get_output().get_node();
        NGRAPH_DEBUG << "Processing (swimming) " << n->get_name();
        if (auto unary = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(n))
        {
            Swimmer nsw{&unary->get_inputs().at(0), csw.reshape};
            work_queue.push_back(nsw);
            NGRAPH_DEBUG << "Propagating reshape " << describe_reshape(csw.reshape) << " for "
                         << n->get_name() << " to " << unary->get_argument(0);
        }
        else if (std::dynamic_pointer_cast<op::Broadcast>(n))
        {
            auto old_broadcast = std::static_pointer_cast<op::Broadcast>(n);
            auto broadcast_axes = old_broadcast->get_broadcast_axes();
            auto broadcast_reshape = csw.reshape;
            bool in_order = true;
            AxisSet new_broadcast_axes;
            std::vector<size_t> new_source_axes;
            auto input_order = broadcast_reshape->get_input_order();
            for (size_t i = 0; i < input_order.size(); i++)
            {
                if (broadcast_axes.count(input_order.at(i)) != 0)
                {
                    new_broadcast_axes.insert(i);
                }
                else
                {
                    if (new_source_axes.size() != 0 && new_source_axes.back() > input_order.at(i))
                    {
                        in_order = false;
                    }
                    new_source_axes.push_back(i);
                }
            }

            auto broadcast_input = old_broadcast->get_argument(0);
            if (!in_order)
            {
                AxisVector new_source_axes_sorted{new_source_axes};
                std::sort(new_source_axes_sorted.begin(), new_source_axes_sorted.end());
                std::map<size_t, size_t> old_new_source_axes;
                for (size_t i = 0; new_source_axes_sorted.size(); i++)
                {
                    old_new_source_axes.insert({new_source_axes.at(i), i});
                }

                AxisVector new_source_axis_order;
                for (auto axis : new_source_axes_sorted)
                {
                    new_source_axis_order.push_back(old_new_source_axes.at(axis));
                }

                auto new_arg_shape =
                    ngraph::apply_permutation(broadcast_input->get_shape(), new_source_axis_order);
                broadcast_input = std::make_shared<op::Reshape>(
                    broadcast_input, new_source_axis_order, new_arg_shape);
            }

            auto new_broadcast = std::make_shared<op::Broadcast>(
                broadcast_input, broadcast_reshape->get_shape(), new_broadcast_axes);
            csw.input->replace_output(new_broadcast->get_outputs().at(0));
        }
        //TODO: Add cases to push through Reshape and BinaryElementwiseArithmetic
        else
        {
            //materialize
            auto new_reshape = csw.reshape->copy_with_new_args({n});
            NGRAPH_DEBUG << "Materializing new reshape " << describe_reshape(new_reshape);
            csw.input->replace_output(new_reshape->get_outputs().at(0));
        }
    }
}

//convert_binary_to_default_order is used when one of the arguments
//of a binary op isn't in the default format (i.e. nhwc instead of nchw)
//We have to normalize this other argument to nchw by swimming nchw towards parameters
//as far as we can
static void convert_binary_to_default_order(
    std::shared_ptr<Node> binary,
    descriptor::Input& input,
    std::shared_ptr<Node> right,
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<op::Reshape>>& reorders,
    std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto left = input.get_output().get_node();
    auto perm_to_def =
        ngraph::get_permutation_to_default_order(reorders.at(right)->get_input_order());
    auto new_shape = apply_permutation(left->get_shape(), perm_to_def);
    NGRAPH_DEBUG << "right = " << ngraph::vector_to_string(right->get_shape()) << ", "
                 << right->get_name();
    auto new_reshape = std::make_shared<op::Reshape>(left, perm_to_def, new_shape);
    NGRAPH_DEBUG << "left : About to swim " << describe_reshape(new_reshape) << " up to "
                 << left->get_name();
    //this should now insert and swim reshape on right
    swim(&input, new_reshape);
    mark_reshape_for_deletion(reorders.at(right), reshapes_to_delete);
    reorders[binary] = reorders.at(right);
}

static void sink_reshape(std::shared_ptr<op::Reshape> reshape,
                         ReshapeMap& reorders,
                         std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto orig_reshape = reorders.at(reshape->get_argument(0));
    if (!reshape->get_is_transpose())
    {
        NGRAPH_DEBUG << "Materializing " << describe_reshape(orig_reshape) << " for reshape "
                     << reshape->get_name();
        insert_reshape(reshape, orig_reshape, 0);
        mark_reshape_for_deletion(orig_reshape, reshapes_to_delete);
        reorders[reshape] = create_default_reshape(reshape);
    }
    else
    {
        //combine both reshapes
        auto new_reshape = combine_reshapes(orig_reshape, reshape);
        //remove original reshape now it's combined with a new one
        //should be safe to remove an already detached node
        mark_reshape_for_deletion(orig_reshape, reshapes_to_delete);
        //replace reshape with combined one
        ngraph::replace_node(reshape, new_reshape);
        reorders[new_reshape] = new_reshape;
        NGRAPH_DEBUG << "Combining " << describe_reshape(orig_reshape) << " and"
                     << describe_reshape(reshape) << " into  " << describe_reshape(new_reshape);
    }
}

static void sink_unary(std::shared_ptr<op::util::UnaryElementwiseArithmetic> n,
                       ReshapeMap& reorders,
                       std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto arg_reshape = reorders.at(n->get_argument(0));
    NGRAPH_DEBUG << "Propagating " << describe_reshape(arg_reshape) << " for " << n->get_name();
    reorders[n] = reorders[n->get_argument(0)];
}

static void sink_binary(std::shared_ptr<op::util::BinaryElementwiseArithmetic> binary,
                        ReshapeMap& reorders,
                        std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto left = binary->get_argument(0);
    auto right = binary->get_argument(1);

    if (reorders.at(left)->get_input_order() == reorders.at(right)->get_input_order())
    {
        NGRAPH_DEBUG << "Propagating " << describe_reshape(reorders.at(left)) << " for "
                     << binary->get_name();
        reorders[binary] = reorders.at(left);
        //at this point, both reshapes will be eventually removed
        mark_reshape_for_deletion(reorders.at(left), reshapes_to_delete);
        mark_reshape_for_deletion(reorders.at(right), reshapes_to_delete);
    }
    else if (reorders.at(left)->get_input_order() == ngraph::get_default_order(left->get_shape()))
    {
        convert_binary_to_default_order(
            binary, binary->get_inputs().at(0), right, reorders, reshapes_to_delete);
    }
    else if (reorders.at(right)->get_input_order() == ngraph::get_default_order(right->get_shape()))
    {
        convert_binary_to_default_order(
            binary, binary->get_inputs().at(1), left, reorders, reshapes_to_delete);
    }
    else
    {
        NGRAPH_DEBUG << "Materializing both reshapes for " << binary->get_name();
        NGRAPH_DEBUG << "Left = " << describe_reshape(reorders.at(left));
        NGRAPH_DEBUG << "Right = " << describe_reshape(reorders.at(right));
        mark_reshape_for_deletion(reorders.at(left), reshapes_to_delete);
        mark_reshape_for_deletion(reorders.at(right), reshapes_to_delete);
        insert_reshape(binary, reorders.at(left), 0);
        insert_reshape(binary, reorders.at(right), 1);
    }
}

static void sink_quantize(std::shared_ptr<op::Quantize> quantize,
                          ReshapeMap& reorders,
                          std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto arg_reshape = reorders.at(quantize->get_argument(0));
    AxisSet axes_in_def_order =
        get_quantization_axes_in_default_order(arg_reshape, quantize->get_axes());
    auto new_quantize = std::make_shared<op::Quantize>(quantize->get_argument(0),
                                                       quantize->get_argument(1),
                                                       quantize->get_argument(2),
                                                       quantize->get_element_type(),
                                                       axes_in_def_order,
                                                       quantize->get_round_mode());

    ngraph::replace_node(quantize, new_quantize);
    reorders[new_quantize] = arg_reshape;
}

static void sink_dequantize(std::shared_ptr<op::Dequantize> dequantize,
                            ReshapeMap& reorders,
                            std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    auto arg_reshape = reorders.at(dequantize->get_argument(0));
    AxisSet axes_in_def_order =
        get_quantization_axes_in_default_order(arg_reshape, dequantize->get_axes());
    auto new_dequantize = std::make_shared<op::Dequantize>(dequantize->get_argument(0),
                                                           dequantize->get_argument(1),
                                                           dequantize->get_argument(2),
                                                           dequantize->get_element_type(),
                                                           axes_in_def_order);

    ngraph::replace_node(dequantize, new_dequantize);
    reorders[new_dequantize] = arg_reshape;
}

static void materialize_shapes(std::shared_ptr<Node> n,
                               ReshapeMap& reorders,
                               std::set<std::shared_ptr<Node>>& reshapes_to_delete)
{
    //skip multiple output nodes and deal with GOEs exclusively
    if (n->get_outputs().size() > 1)
    {
        return;
    }

    for (size_t i = 0; i < n->get_arguments().size(); i++)
    {
        //materialize all pending reshapes, flush pending reshapes
        auto arg = n->get_argument(i);
        if (reorders.count(arg) != 0)
        {
            NGRAPH_DEBUG << "Materializing " << describe_reshape(reorders.at(arg)) << " for "
                         << arg->get_name();
            mark_reshape_for_deletion(reorders.at(arg), reshapes_to_delete);
            insert_reshape(n, reorders.at(arg), i);
            //no swimming up
        }
    }
    reorders[n] = create_default_reshape(n);
}

//The goal of ReshapeSinking is to remove
//round-trip reshapes(i.e. nhwc->nchw(nchw-only-op)->nhwc)
//around nchw-only-op (e.g.Convolution, Batchnorm, Avg/MaxPool)
//This is achieved by both **sinking**, propagating reshapes
//through ops towards op::Results,
//or **swimming** Reshapes up towards op::Parameter
//For each op type we support we can either combine
//two reshapes by replacing the existing Reshape,
//materialize pending reshapes if they can't be propagated through op
bool ngraph::pass::ReshapeSinking::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    ReshapeMap reorders;
    NodeVector results;
    std::set<std::shared_ptr<Node>> reshapes_to_delete;

    //STEP 1 : Sink or Swim reshapes away for op clusters
    for (auto n : f->get_ordered_ops())
    {
        NGRAPH_DEBUG << "Processing node " << n->get_name();
        //collect all Result nodes for a sanity check
        if (n->is_output())
        {
            results.push_back(n);
        }

        if (auto reshape = std::dynamic_pointer_cast<op::Reshape>(n))
        {
            sink_reshape(reshape, reorders, reshapes_to_delete);
        }
        else if (auto unary = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(n))
        {
            sink_unary(unary, reorders, reshapes_to_delete);
        }
        else if (auto binary = std::dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(n))
        {
            sink_binary(binary, reorders, reshapes_to_delete);
        }
        else if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(n))
        {
            reorders[goe] = create_default_reshape(goe);
        }
        else if (auto quantize = std::dynamic_pointer_cast<op::Quantize>(n))
        {
            sink_quantize(quantize, reorders, reshapes_to_delete);
        }
        else if (auto dequantize = std::dynamic_pointer_cast<op::Dequantize>(n))
        {
            sink_dequantize(dequantize, reorders, reshapes_to_delete);
        }
        else
        {
            materialize_shapes(n, reorders, reshapes_to_delete);
        }
    }

    //STEP 2: purge all the reshapes we either sunk or swam.
    for (auto r : reshapes_to_delete)
    {
        delete_reshape(r);
    }

    //make sure shapes are always materialized before results
    for (auto r : results)
    {
        NGRAPH_ASSERT(r->get_shape() == r->get_argument(0)->get_shape() &&
                      r->get_element_type() == r->get_argument(0)->get_element_type())
            << " op::Result = " << *r << ", Arg = " << *r->get_argument(0);
    }

    //STEP 3: fix wrong shape info wholesale
    for (auto n : f->get_ordered_ops())
    {
        n->revalidate_and_infer_types();
    }
    return true;
}
