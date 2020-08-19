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

#include "constant_folding.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/reduce_logical_and.hpp"
#include "ngraph/op/reduce_logical_or.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/any.hpp"

using namespace std;
using namespace ngraph;

static Shape get_shape_no_keep_dims(const AxisSet& reduction_axes, const Shape& input_shape)
{
    Shape shape_no_keep_dims;

    for (size_t i = 0; i < input_shape.size(); i++)
    {
        if (reduction_axes.count(i) == 0)
        {
            shape_no_keep_dims.push_back(input_shape[i]);
        }
    }

    return shape_no_keep_dims;
}

static Output<Node> fold_constant_logical_reduction(shared_ptr<op::v0::Constant> constant,
                                                    shared_ptr<Node> reduction_node)
{
    runtime::AlignedBuffer buffer(shape_size(reduction_node->get_output_shape(0)) * sizeof(char));
    char* data_ptr = buffer.get_ptr<char>();

    if (auto all = as_type_ptr<::ngraph::op::v0::All>(reduction_node))
    {
        runtime::reference::all(constant->get_vector<char>().data(),
                                data_ptr,
                                constant->get_output_shape(0),
                                reduction_node->get_output_shape(0),
                                all->get_reduction_axes());
    }
    else if (auto any = as_type_ptr<::ngraph::op::v0::Any>(reduction_node))
    {
        runtime::reference::any(constant->get_vector<char>().data(),
                                data_ptr,
                                constant->get_output_shape(0),
                                reduction_node->get_output_shape(0),
                                any->get_reduction_axes());
    }
    else if (auto reduce_and = as_type_ptr<::ngraph::op::v1::ReduceLogicalAnd>(reduction_node))
    {
        const auto reduction_axes = reduce_and->get_reduction_axes();
        const auto input_shape = reduce_and->get_input_shape(0);

        runtime::reference::all(constant->get_vector<char>().data(),
                                data_ptr,
                                constant->get_output_shape(0),
                                get_shape_no_keep_dims(reduction_axes, input_shape),
                                reduction_axes);
    }
    else if (auto reduce_or = as_type_ptr<::ngraph::op::v1::ReduceLogicalOr>(reduction_node))
    {
        const auto reduction_axes = reduce_or->get_reduction_axes();
        const auto input_shape = reduce_or->get_input_shape(0);

        runtime::reference::any(constant->get_vector<char>().data(),
                                data_ptr,
                                constant->get_output_shape(0),
                                get_shape_no_keep_dims(reduction_axes, input_shape),
                                reduction_axes);
    }
    else
    {
        NGRAPH_CHECK(false,
                     "Internal nGraph error: Ops handled in "
                     "fold_constant_logical_reduction must be consistent with those "
                     "matched in construct_constant_logical_reduction");
    }

    return make_shared<op::v0::Constant>(reduction_node->get_output_element_type(0),
                                         reduction_node->get_output_shape(0),
                                         data_ptr)
        ->output(0);
}

void pass::ConstantFolding::construct_constant_logical_reduction()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::v0::Constant>());
    auto constant_axes_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2}, pattern::has_class<op::v0::Constant>());
    auto is_supported_reduction = [](Output<Node> n) {
        return (pattern::has_class<::ngraph::op::v0::All>()(n) ||
                pattern::has_class<::ngraph::op::v0::Any>()(n) ||
                pattern::has_class<::ngraph::op::v1::ReduceLogicalAnd>()(n) ||
                pattern::has_class<::ngraph::op::v1::ReduceLogicalOr>()(n));
    };
    auto reduction =
        std::make_shared<pattern::op::Any>(element::i32,
                                           Shape{2},
                                           is_supported_reduction,
                                           OutputVector{constant_data_label, constant_axes_label});

    auto constant_logical_reduction_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_logical_reduction_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match =
            static_pointer_cast<op::v0::Constant>(pattern_map[constant_data_label]);
        Output<Node> reduction_match = m.get_match_value();

        NGRAPH_CHECK(revalidate_and_ensure_static(reduction_match.get_node_shared_ptr()));

        reduction_match.replace(
            fold_constant_logical_reduction(constant_match, reduction_match.get_node_shared_ptr()));
        return true;
    };

    auto logical_reduction_matcher =
        make_shared<pattern::Matcher>(reduction, "ConstantFolding.ConstantLogicalReduction");
    this->add_matcher(logical_reduction_matcher,
                      constant_logical_reduction_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}
