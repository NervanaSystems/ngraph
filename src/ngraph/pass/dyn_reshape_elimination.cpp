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

#include "dyn_reshape_elimination.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/dyn_reshape.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

void ngraph::pass::DynReshapeElimination::construct_dyn_reshape_pattern()
{
    Shape shape_data_op{1, 2, 3};
    Shape shape_shape_op{2};

    auto data_op = std::make_shared<pattern::op::Label>(element::f32, shape_data_op);
    auto shape_op = std::make_shared<pattern::op::Label>(element::i64, shape_shape_op);
    auto dyn_reshape = std::make_shared<op::DynReshape>(data_op, shape_op);

    auto callback = [data_op, shape_op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_dyn_reshape_pattern against node = "
                     << *(m.get_match_root());
        auto pattern_map = m.get_pattern_map();
        auto dop = pattern_map[data_op];
        auto sop = pattern_map[shape_op];

        if (!sop->get_outputs()[0].has_static_value())
        {
            NGRAPH_DEBUG << "Bailing, because no static value on the shape input";
        }

        AxisVector perm;
        for (size_t i = 0; i < dop->get_shape().size(); i++)
        {
            perm.push_back(i);
        }
        auto replacement =
            std::make_shared<ngraph::op::Reshape>(dop, perm, sop->get_output_static_value(0));

        NGRAPH_DEBUG << "Replacing " << *(m.get_match_root()) << " with " << *replacement;

        ngraph::replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(dyn_reshape, callback);
    this->add_matcher(m);
}
