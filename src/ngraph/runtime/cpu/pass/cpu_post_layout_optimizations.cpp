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

#include <algorithm>
#include <typeindex>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"

using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

void ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations::construct_weight_fusion()
{
    auto param = std::make_shared<pattern::op::Label>(element::f32, Shape{64});
    auto reshape_conv =
        std::make_shared<ngraph::op::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto data_conv = std::make_shared<pattern::op::Label>(element::f32, Shape{16, 4, 7, 7});
    auto tvt = reshape_conv->get_outputs().at(0).get_tensor_ptr().get();
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt);
    auto cvt_lt_conv = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv, lt_desc);
    auto conv = std::make_shared<ngraph::op::Convolution>(
        data_conv, cvt_lt_conv, Strides{1, 1}, Strides{1, 1});

    pattern::graph_rewrite_callback callback = [param](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_weight against "
                     << m.get_match_root()->get_name();

        auto m_cvt_lt = m.get_match_root()->get_argument(1);
        auto m_reshape_conv = m_cvt_lt->get_argument(0);

        std::shared_ptr<Node> m_conv_bprop;

        std::vector<std::type_index> user_pattern = {TI(ngraph::op::Reshape),
                                                     TI(runtime::cpu::op::ConvertLayout),
                                                     TI(ngraph::op::ConvolutionBackpropData)};

        for (auto u : m.get_pattern_map()[param]->get_users())
        {
            if (u != m_reshape_conv)
            {
                size_t num_matches = 0;
                auto ui = u;
                for (; num_matches < user_pattern.size(); num_matches++)
                {
                    const Node& user_ref = *ui;
                    if (TI(user_ref) != user_pattern.at(num_matches))
                    {
                        NGRAPH_DEBUG << "the type for user " << ui->get_name()
                                     << " doesn't match the type at " << num_matches;
                        break;
                    }

                    if (ui->get_users().size() != 1)
                    {
                        NGRAPH_DEBUG << u->get_name() << " has more than one user";
                        break;
                    }
                    ui = ui->get_users().at(0);
                }

                if (num_matches == user_pattern.size())
                {
                    m_conv_bprop = u->get_users().at(0)->get_users().at(0);
                    NGRAPH_DEBUG << " m_conv_bprop is set to " << m_conv_bprop->get_name();
                    break;
                }
            }
        }

        if (!m_conv_bprop)
        {
            return false;
        }

        auto m_cvt_lt_bprop = m_conv_bprop->get_argument(0);
        auto m_reshape_bprop = m_cvt_lt_bprop->get_argument(0);

        NGRAPH_DEBUG << "Replacing input "
                     << m_cvt_lt_bprop->get_inputs().at(0).get_output().get_node()->get_name()
                     << " to " << m_cvt_lt_bprop->get_name() << " with "
                     << m_cvt_lt->get_outputs().at(0).get_node()->get_name();
        m_cvt_lt_bprop->get_inputs().at(0).replace_output(m_cvt_lt->get_outputs().at(0));

        return true;
    };

    auto m = make_shared<pattern::Matcher>(conv, callback);
    this->add_matcher(m);
}
