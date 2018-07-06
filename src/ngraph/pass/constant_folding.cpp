/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <stdint.h>

#include "constant_folding.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> make_constant_reshape(shared_ptr<op::Constant> constant,
                                               shared_ptr<op::Reshape> reshape)
{
    auto out_shape = reshape->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::reshape<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_shape(),
                                   reshape->get_input_order(),
                                   out_shape);

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void ngraph::pass::ConstantFolding::construct_constant_reshape()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto reshape = make_shared<op::Reshape>(constant_label, AxisVector{0, 1}, Shape{2, 4, 1});

    auto constant_reshape_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reshape_match = dynamic_pointer_cast<op::Reshape>(m.get_match_root());

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<int>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<int8_t>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<float>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<double>(constant_match, reshape_match));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(reshape, constant_reshape_callback);
    this->add_matcher(reshape_matcher);
}

template <class T>
shared_ptr<op::Constant> make_constant_broadcast(shared_ptr<op::Constant> constant,
                                                 shared_ptr<op::Broadcast> broadcast)
{
    auto out_shape = broadcast->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::broadcast<T>(constant->get_vector<T>().data(),
                                     out_vec.data(),
                                     constant->get_shape(),
                                     out_shape,
                                     broadcast->get_broadcast_axes());

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void ngraph::pass::ConstantFolding::construct_constant_broadcast()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());

    auto broadcast = make_shared<op::Broadcast>(constant_label, Shape{2, 4}, AxisSet{1});

    auto constant_broadcast_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto broadcast_match = dynamic_pointer_cast<op::Broadcast>(m.get_match_root());

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<int>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<int8_t>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<float>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<double>(constant_match, broadcast_match));
            return true;
        }

        return false;
    };

    auto broadcast_matcher = make_shared<pattern::Matcher>(broadcast, constant_broadcast_callback);
    this->add_matcher(broadcast_matcher);
}
