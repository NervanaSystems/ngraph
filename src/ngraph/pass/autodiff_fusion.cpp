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
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/autodiff_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace ngraph;
using namespace std;

void pass::AutodiffFusion::construct_cross_entropy_softmax()
{
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 4});
    auto labels = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 4});
    auto softmax = make_shared<op::Softmax>(input, AxisSet{1});
    auto sm_log = make_shared<op::Log>(input);
    auto sum_ce = make_shared<op::Sum>(labels * sm_log, AxisSet{1});

    pattern::graph_rewrite_callback callback = [input, labels](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for AutodiffFusion::construct_cross_entropy_softmax against "
                     << m.match_root()->get_name();

        auto m_softmax = std::dynamic_pointer_cast<op::Softmax>(m.match_root()->get_input_op(0));

        auto pattern_map = m.get_pattern_map();
        auto m_input = m.get_pattern_map()[input];

        if (m_input->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << m_input->get_name() << "'s number of dimensions isn't equal to 2";
            return false;
        }

        if (m_softmax->get_axes().size() != 1)
        {
            NGRAPH_DEBUG << "number of reduction axes for softmax " << m_softmax->get_name()
                         << "isn't equal to 1";
            return false;
        }

        size_t axis = *(m_softmax->get_axes().begin());
        auto cross_entropy =
            std::make_shared<op::CrossEntropySoftMax>(m_input, labels, m.match_root(), axis);
        ngraph::replace_node(m.match_root(), cross_entropy);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sum_ce, callback);
    this->add_matcher(m);
}
