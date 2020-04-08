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
#include "ngraph/node.hpp"
#include "ngraph/op/util/value_propagarion.hpp"
#include "ngraph/pattern/op/pattern.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void pass::ConstantFolding::construct_constant_default()
{
    auto any_op = make_shared<pattern::op::Label>(
        element::f32, Shape{1}, [](const std::shared_ptr<Node>& value) { return true; });

    auto constant_default_callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_default_callback against node = "
                     << m.get_match_root()->get_name();

        for (auto& input : m.get_match_root()->inputs())
        {
            if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                    input.get_source_output().get_node_shared_ptr()))
            {
                return false;
            }
        }

        if (auto node = std::dynamic_pointer_cast<op::util::ValuePropagation>(m.get_match_root()))
        {
            auto replacement = node->fold_constant(m.get_match_root());
            replace_node(m.get_match_root(), replacement);
            return true;
        }

        return false;
    };

    auto default_matcher = make_shared<pattern::Matcher>(any_op, "ConstantFoldingDefault");
    this->add_matcher(
        default_matcher, constant_default_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
