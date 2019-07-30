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

#include <algorithm>
#include <iterator>

#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/clamp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

// Modify input vector in-place and return reference to modified vector.
static vector<string> to_lower_case(const vector<string>& vs)
{
    vector<string> res(vs);
    transform(begin(res), end(res), begin(res), [](string& s) { return to_lower(s); });
    return res;
}

op::util::RNNCellBase::RNNCellBase(size_t hidden_size,
                                   float clip,
                                   const vector<string>& activations,
                                   const vector<float>& activation_alpha,
                                   const vector<float>& activation_beta)
    : m_hidden_size(hidden_size)
    , m_clip(clip)
    , m_activations(to_lower_case(activations))
    , m_activation_alpha(activation_alpha)
    , m_activation_beta(activation_beta)
{
}

op::util::ActivationFunction op::util::RNNCellBase::get_activation_function(size_t idx) const
{
    op::util::ActivationFunction afunc = get_activation_func_by_name(m_activations.at(idx));

    // Set activation functions parameters (if any)
    if (m_activation_alpha.size() > idx)
    {
        afunc.set_alpha(m_activation_alpha.at(idx));
    }
    if (m_activation_beta.size() > idx)
    {
        afunc.set_beta(m_activation_beta.at(idx));
    }

    return afunc;
}

shared_ptr<Node> op::util::RNNCellBase::add(const Output<Node>& lhs, const Output<Node>& rhs)
{
    auto args = op::numpy_style_broadcast_values({lhs, rhs});
    return {make_shared<op::Add>(args.at(0), args.at(1))};
}

shared_ptr<Node> op::util::RNNCellBase::sub(const Output<Node>& lhs, const Output<Node>& rhs)
{
    auto args = op::numpy_style_broadcast_values({lhs, rhs});
    return {make_shared<op::Subtract>(args.at(0), args.at(1))};
}

shared_ptr<Node> op::util::RNNCellBase::mul(const Output<Node>& lhs, const Output<Node>& rhs)
{
    auto args = op::numpy_style_broadcast_values({lhs, rhs});
    return {make_shared<op::Multiply>(args.at(0), args.at(1))};
}

shared_ptr<Node> op::util::RNNCellBase::clip(const Output<Node>& data) const
{
    if (m_clip == 0.f)
    {
        return data.as_single_output_node();
    }

    return make_shared<op::Clamp>(data, -m_clip, m_clip);
}
