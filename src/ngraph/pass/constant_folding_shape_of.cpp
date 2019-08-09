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
#include "ngraph/op/experimental/shape_of.hpp"

using namespace std;
using namespace ngraph;

// ShapeOf is a bit of an odd duck: it doesn't matter if the input's value is
// constant, as long as it has static shape.
void pass::ConstantFolding::construct_constant_shape_of()
{
    auto arg_label = make_shared<pattern::op::Label>(element::i32, Shape{2, 3, 4});
    auto shape_of_op = make_shared<op::ShapeOf>(arg_label);

    auto constant_shape_of_callback = [arg_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_shape_of_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto arg_match = pattern_map[arg_label];

        if (arg_match->get_output_partial_shape(0).is_static())
        {
            NGRAPH_CHECK(revalidate_and_ensure_static(m.get_match_root()));

            auto arg_shape = arg_match->get_output_shape(0);
            auto replacement =
                make_shared<op::Constant>(element::i64, Shape{arg_shape.size()}, arg_shape.data());

            replace_node(m.get_match_root(), replacement);

            return true;
        }
        else
        {
            return false;
        }
    };

    auto shape_of_matcher =
        make_shared<pattern::Matcher>(shape_of_op, "ConstantFolding.ConstantShapeOf");
    this->add_matcher(
        shape_of_matcher, constant_shape_of_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
