//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "transpose_elimination.hpp"

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace ngraph;

void pass::TransposeElimination::construct_transpose_pattern()
{
    Shape shape_op{1, 2};
    auto op = std::make_shared<pattern::op::Label>(element::f32, shape_op);
    auto const_perm =
        std::make_shared<op::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
    auto transpose = std::make_shared<op::Transpose>(op, const_perm);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_transpose_pattern against node = "
                     << m.get_match_root()->get_name();
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}
