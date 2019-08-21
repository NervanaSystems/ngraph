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
#include <typeindex>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
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

    auto callback = [param](pattern::Matcher& m) {
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

    auto m =
        make_shared<pattern::Matcher>(conv, "CPUPostLayoutOptimizations.ConstructWeight_fusion");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations::construct_slice_convertLayout_fusion()
{
    auto param = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 576, 17, 17});
    auto slice = std::make_shared<ngraph::op::Slice>(
        param, Coordinate{0, 0, 0, 0}, Coordinate{1, 192, 17, 17});
    auto tvt = slice->get_outputs().at(0).get_tensor_ptr().get();
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt);
    auto cvt_lt = std::make_shared<runtime::cpu::op::ConvertLayout>(slice, lt_desc);

    auto callback = [param](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_slice_converLayout against "
                     << m.get_match_root()->get_name();

        auto m_cvt_lt = m.get_match_root();
        auto m_slice = m_cvt_lt->get_argument(0);
        auto slice_ptr = static_cast<const ngraph::op::Slice*>(m_slice.get());
        // do the fusion if slice has 1 user and uses mkldnn kernel.
        if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(slice_ptr) ||
            m_slice->get_users().size() != 1)
        {
            return false;
        }

        for (auto u : m.get_pattern_map()[param]->get_users())
        {
            if (u != m_slice)
            {
                continue;
            }

            auto new_slice = std::make_shared<ngraph::op::Slice>(m_slice->get_argument(0),
                                                                 slice_ptr->get_lower_bounds(),
                                                                 slice_ptr->get_upper_bounds(),
                                                                 slice_ptr->get_strides());
            auto op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
            op_annotations->set_mkldnn_op(true);
            new_slice->set_op_annotations(op_annotations);
            auto tv = new_slice->get_output_tensor_ptr(0);
            auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
            layout->set_mkldnn_md(mkldnn_utils::get_output_mkldnn_md(m_cvt_lt.get(), 0));
            tv->set_tensor_layout(layout);
            ngraph::replace_node(m_cvt_lt, new_slice);
        }

        return true;
    };

    auto m = make_shared<pattern::Matcher>(
        cvt_lt, "CPUPostLayoutOptimizations.ConstructSliceConvertLayoutFusion");
    this->add_matcher(m, callback);
}

// Reshape(transpose) + ConvertLayout
// MKLDNN has more efficient ConvertLayout kernels for named/non-padded formats
// If a transpose is converting a padded format into a generic padded/blocked format, it is better
// to ConvertLayout first and then do the transpose
// E.g.,
// Shape{10, 20, 30, 40} -(Reshape)-> Shape{10, 40, 20, 30} -(ConvertLayout)-> Shape{10, 40, 20, 30}
// is changed to
// Shape{10, 20, 30, 40} -(ConvertLayout)-> Shape{10, 20, 30, 40} -(Reshape)-> Shape{10, 40, 20, 30}
// The new ConvertLayout op computes the desired output layout (out_md) directly from
// input layout using a rotated out_md
void ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations::
    construct_reshape_convertLayout_fusion()
{
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto reshape =
        std::make_shared<ngraph::op::Reshape>(input, AxisVector{0, 1, 2, 3}, Shape{1, 1, 1, 1});
    auto lt_desc =
        std::make_shared<runtime::cpu::LayoutDescriptor>(*reshape->get_output_tensor_ptr());
    auto cvt_lt = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape, lt_desc);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_reshape_converLayout against "
                     << m.get_match_root()->get_name();

        auto cvt_lt_m = static_pointer_cast<runtime::cpu::op::ConvertLayout>(m.get_match_root());
        auto reshape_m = static_pointer_cast<ngraph::op::Reshape>(cvt_lt_m->get_argument(0));

        if (reshape_m->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "ReshapeConvertLayout: Reshape has multiple users";
            return false;
        }

        if (!reshape_m->get_is_transpose())
        {
            NGRAPH_DEBUG << "ReshapeConvertLayout: Reshape is not a transpose";
            return false;
        }

        auto annotation = reshape_m->get_op_annotations();
        if (!annotation || annotation->get_in_place_oi_pairs().size() == 0)
        {
            NGRAPH_DEBUG << "ReshapeConvertLayout: Reshape is not pass-through";
            return false;
        }

        auto reshape_m_md = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(reshape_m.get(), 0);
        if (reshape_m_md.data.format != mkldnn_blocked ||
            !runtime::cpu::mkldnn_utils::is_mkldnn_padded_layout(
                reshape_m_md, ngraph::get_default_order(reshape_m->get_shape())))
        {
            NGRAPH_DEBUG << "ReshapeConvertLayout: Reshape is not creating a blocked/padded layout";
            return false;
        }

        // Rotate output layout to the pre-transposed order
        auto out_md = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(cvt_lt_m.get(), 0);
        auto reshape_order = reshape_m->get_input_order();
        // Get the inverse of the original transpose order
        // E.g., [0, 3, 1, 2] -> [0, 2, 3, 1]
        AxisVector inverse_order;
        for (int i = 0; i < reshape_order.size(); i++)
        {
            inverse_order.push_back(std::find(reshape_order.begin(), reshape_order.end(), i) -
                                    reshape_order.begin());
        }
        auto rotated_md = runtime::cpu::mkldnn_utils::rotate_blocked_md(out_md, inverse_order);
        auto rotated_lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(
            *reshape_m->get_argument(0)->get_output_tensor_ptr());
        rotated_lt_desc->set_mkldnn_md(rotated_md);

        auto cvt_lt_n = std::make_shared<runtime::cpu::op::ConvertLayout>(
            reshape_m->get_argument(0), 0, rotated_lt_desc);
        cvt_lt_n->set_op_annotations(cvt_lt_m->get_op_annotations());

        auto reshape_n =
            std::make_shared<ngraph::op::Reshape>(cvt_lt_n, reshape_order, cvt_lt_m->get_shape());
        auto reshape_n_layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(
            *reshape_n->get_output_tensor_ptr());
        reshape_n_layout->set_mkldnn_md(out_md);
        reshape_n->get_output_tensor_ptr()->set_tensor_layout(reshape_n_layout);
        reshape_n->set_op_annotations(reshape_m->get_op_annotations());

        ngraph::replace_node(cvt_lt_m, reshape_n);
        NGRAPH_DEBUG << "ReshapeConvertLayout: Reordering reshape and convertlayout for faster "
                        "MKLDNN kernels";

        return true;
    };

    auto m = make_shared<pattern::Matcher>(
        cvt_lt, "CPUPostLayoutOptimizations.ConstructReshapeConvertLayoutFusion");
    this->add_matcher(m, callback);
}
