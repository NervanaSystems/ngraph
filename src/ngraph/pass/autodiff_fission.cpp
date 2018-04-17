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
#include <unordered_set>

#include "ngraph/pass/core_fusion.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/cross_entropy_softmax.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/pass/autodiff_fission.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace ngraph;
using namespace std;

void pass::AutodiffFission::construct_cross_entropy_softmax()
{
    auto ce_sm_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::CrossEntropySoftMax>(n));
    };

    auto ce_sm = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 4}, ce_sm_pred);

    pattern::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG
            << "In a callback for AutodiffFission::construct_cross_entropy_softmax against "
            << m.match_root()->get_name();

        auto m_ce_sm = std::dynamic_pointer_cast<op::CrossEntropySoftMax>(m.match_root());
        auto old_ce = m_ce_sm->get_original_cross_entropy();
        ngraph::replace_node(m.match_root(), old_ce);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(ce_sm, callback);
    this->add_matcher(m);
}
