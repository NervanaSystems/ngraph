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

#include "dyn_elimination.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace std;
using namespace ngraph;

pass::DynElimination::DynElimination()
    : GraphRewrite()
{
    construct_transpose();
}

void pass::DynElimination::construct_transpose()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto perm_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());

    auto transpose = make_shared<op::Transpose>(data_arg_label, perm_arg_label);

    auto transpose_callback = [data_arg_label, perm_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto data_arg = pattern_map[data_arg_label];
        auto perm_arg = static_pointer_cast<op::Constant>(pattern_map[perm_arg_label]);

        // TODO(amprocte): Can't handle the case where data shape is dynamic, because static
        // Reshape requries the exact output shape to be declared. See if we can come up with a
        // workaround.
        if (data_arg->get_output_partial_shape(0).is_dynamic())
        {
            return false;
        }

        auto& data_shape = data_arg->get_output_shape(0);

        // TODO(amprocte): These should be redundant if the graph is validated. Necessary?
        if (perm_arg->get_element_type() != element::i64 ||
            perm_arg->get_output_partial_shape(0).is_dynamic() ||
            perm_arg->get_output_shape(0).size() != 1)
        {
            return false;
        }

        auto perm = perm_arg->get_axis_vector_val();

        auto output_shape = ngraph::apply_permutation(data_shape, perm);

        auto replacement = std::make_shared<op::Reshape>(data_arg, perm, output_shape);

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto transpose_matcher = make_shared<pattern::Matcher>(transpose, "DynElimination.Transpose");
    add_matcher(transpose_matcher, transpose_callback, all_pass_property_off);
}
