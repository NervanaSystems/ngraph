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

#include "constant_folding.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/any.hpp"

using namespace std;
using namespace ngraph;

static shared_ptr<op::Constant> fold_constant_logical_reduction(shared_ptr<op::Constant> constant,
                                                                shared_ptr<Node> reduction_node)
{
    vector<char> out_vec(shape_size(reduction_node->get_shape()));

    if (auto all = dynamic_pointer_cast<::ngraph::op::All>(reduction_node))
    {
        runtime::reference::all(constant->get_vector<char>().data(),
                                out_vec.data(),
                                constant->get_output_shape(0),
                                reduction_node->get_shape(),
                                all->get_reduction_axes());
    }
    else if (auto any = dynamic_pointer_cast<::ngraph::op::Any>(reduction_node))
    {
        runtime::reference::any(constant->get_vector<char>().data(),
                                out_vec.data(),
                                constant->get_output_shape(0),
                                reduction_node->get_shape(),
                                any->get_reduction_axes());
    }
    else
    {
        NGRAPH_CHECK(false,
                     "Internal nGraph error: Ops handled in "
                     "fold_constant_logical_reduction must be consistent with those "
                     "matched in construct_constant_logical_reduction");
    }

    return make_shared<op::Constant>(
        reduction_node->get_output_element_type(0), reduction_node->get_shape(), out_vec);
}

void pass::ConstantFolding::construct_constant_logical_reduction()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto is_supported_reduction = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<::ngraph::op::All>()(n) ||
                pattern::has_class<::ngraph::op::Any>()(n));
    };
    auto reduction =
        std::make_shared<pattern::op::Any>(element::i32,
                                           Shape{2},
                                           is_supported_reduction,
                                           NodeVector{constant_data_label, constant_axes_label});

    auto constant_logical_reduction_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_logical_reduction_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto reduction_match = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(reduction_match));

        replace_node(reduction_match,
                     fold_constant_logical_reduction(constant_match, reduction_match));
        return true;
    };

    auto logical_reduction_matcher =
        make_shared<pattern::Matcher>(reduction, "ConstantFolding.ConstantLogicalReduction");
    this->add_matcher(logical_reduction_matcher,
                      constant_logical_reduction_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}
