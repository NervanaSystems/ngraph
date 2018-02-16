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

#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "ngraph/pattern/core_fusion.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace ngraph;
using namespace std;

bool is_zero(shared_ptr<Node> reduce_constant)
{
    auto result_bool = is_equal_to_const_value("0", reduce_constant);
    return result_bool;
}

static shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::f32, Shape{}, {n});
}

void pass::CoreFusion::construct_relu_pattern()
{
    auto iconst0 = construct_constant_node(0);
    auto val = make_shared<pattern::op::Label>(iconst0);
    auto zero = make_shared<pattern::op::Label>(iconst0, nullptr, Nodes{iconst0});
    auto max = make_shared<op::Maximum>(zero, val);

    pattern::gr_callback_fn callback = [val, zero](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_relu_pattern against "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        shared_ptr<Node> nn;

        auto mzero = m.get_pattern_map()[zero];
        if (!is_zero(mzero))
        {
            cout << "zero constant = " << mzero->get_name() << " not equal to 0\n";
            return nn;
        }
        auto mpattern = m.match_root();

        auto cg = shared_ptr<Node>(new op::Relu(pattern_map[val]));
        return cg;
    };

    auto m = make_shared<pattern::Matcher>(max, callback);
    this->add_matcher(m);
}
