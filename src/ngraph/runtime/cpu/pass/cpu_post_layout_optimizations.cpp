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

#include <algorithm>
#include <typeindex>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

void ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations::construct_weight_fusion()
{
    auto param = std::make_shared<pattern::op::Label>(element::f32, Shape{64});
    auto reshape_conv =
        std::make_shared<ngraph::op::v0::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto data_conv = std::make_shared<pattern::op::Label>(element::f32, Shape{16, 4, 7, 7});
    descriptor::Tensor& tvt = reshape_conv->get_output_tensor(0);
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(tvt);
    auto cvt_lt_conv = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv, lt_desc);
    auto conv = std::make_shared<ngraph::op::v0::Convolution>(
        data_conv, cvt_lt_conv, Strides{1, 1}, Strides{1, 1});

    auto callback = [param](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_weight against "
                     << m.get_match_root()->get_name();

        auto m_cvt_lt = m.get_match_value().get_node()->get_argument(1);
        auto m_reshape_conv = m_cvt_lt->get_argument(0);

        std::shared_ptr<Node> m_conv_bprop;

        std::vector<std::type_index> user_pattern = {TI(ngraph::op::v0::Reshape),
                                                     TI(runtime::cpu::op::ConvertLayout),
                                                     TI(ngraph::op::v0::ConvolutionBackpropData)};

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

        NGRAPH_DEBUG << "Replacing input " << m_cvt_lt_bprop->get_argument(0)->get_name() << " to "
                     << m_cvt_lt_bprop->get_name() << " with " << m_cvt_lt->get_name();
        m_cvt_lt_bprop->input(0).replace_source_output(m_cvt_lt->output(0));

        return true;
    };

    auto m =
        make_shared<pattern::Matcher>(conv, "CPUPostLayoutOptimizations.ConstructWeight_fusion");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations::construct_slice_convertLayout_fusion()
{
    auto param = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 576, 17, 17});
    auto slice = std::make_shared<ngraph::op::v0::Slice>(
        param, Coordinate{0, 0, 0, 0}, Coordinate{1, 192, 17, 17});
    descriptor::Tensor& tvt = slice->get_output_tensor(0);
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(tvt);
    auto cvt_lt = std::make_shared<runtime::cpu::op::ConvertLayout>(slice, lt_desc);

    auto callback = [param](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_slice_converLayout against "
                     << m.get_match_root()->get_name();

        auto m_cvt_lt = m.get_match_value();
        auto m_slice = m_cvt_lt.get_node()->get_argument(0);
        auto slice_ptr = static_cast<const ngraph::op::v0::Slice*>(m_slice.get());
        // do the fusion if slice has 1 user and uses dnnl kernel.
        if (!runtime::cpu::dnnl_utils::use_dnnl_kernel(slice_ptr) ||
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

            auto new_slice = std::make_shared<ngraph::op::v0::Slice>(m_slice->input_value(0),
                                                                     slice_ptr->get_lower_bounds(),
                                                                     slice_ptr->get_upper_bounds(),
                                                                     slice_ptr->get_strides());
            auto op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
            op_annotations->set_dnnl_op(true);
            new_slice->set_op_annotations(op_annotations);
            auto tv = new_slice->get_output_tensor_ptr(0);
            auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
            layout->set_dnnl_md(dnnl_utils::get_output_dnnl_md(m_cvt_lt.get_node(), 0));
            tv->set_tensor_layout(layout);
            m_cvt_lt.replace(new_slice->output(0));
        }

        return true;
    };

    auto m = make_shared<pattern::Matcher>(
        cvt_lt, "CPUPostLayoutOptimizations.ConstructSliceConvertLayoutFusion");
    this->add_matcher(m, callback);
}

// Reshape(transpose) + ConvertLayout
// DNNL has more efficient ConvertLayout kernels for named/non-padded formats
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
        std::make_shared<ngraph::op::v0::Reshape>(input, AxisVector{0, 1, 2, 3}, Shape{1, 1, 1, 1});
    auto lt_desc =
        std::make_shared<runtime::cpu::LayoutDescriptor>(*reshape->get_output_tensor_ptr(0));
    auto cvt_lt = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape, lt_desc);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_reshape_converLayout against "
                     << m.get_match_root()->get_name();

        auto cvt_lt_m = m.get_match_root_as<runtime::cpu::op::ConvertLayout>();
        NGRAPH_CHECK(cvt_lt_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `runtime::cpu::op::ConvertLayout`");
        auto reshape_m = static_pointer_cast<ngraph::op::v0::Reshape>(cvt_lt_m->get_argument(0));

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

        auto reshape_m_md = runtime::cpu::dnnl_utils::get_output_dnnl_md(reshape_m.get(), 0);
        if (reshape_m_md.data.FORMAT_KIND != dnnl_blocked ||
            !runtime::cpu::dnnl_utils::is_dnnl_padded_layout(
                reshape_m_md, ngraph::get_default_order(reshape_m->get_output_shape(0))))
        {
            NGRAPH_DEBUG << "ReshapeConvertLayout: Reshape is not creating a blocked/padded layout";
            return false;
        }

        // Rotate output layout to the pre-transposed order
        auto out_md = runtime::cpu::dnnl_utils::get_output_dnnl_md(cvt_lt_m.get(), 0);
        auto reshape_order = reshape_m->get_input_order();
        // Get the inverse of the original transpose order
        // E.g., [0, 3, 1, 2] -> [0, 2, 3, 1]
        AxisVector inverse_order;
        for (size_t i = 0; i < reshape_order.size(); i++)
        {
            inverse_order.push_back(std::find(reshape_order.begin(), reshape_order.end(), i) -
                                    reshape_order.begin());
        }
        auto rotated_md = runtime::cpu::dnnl_utils::rotate_blocked_md(out_md, inverse_order);
        auto rotated_lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(
            *reshape_m->get_argument(0)->get_output_tensor_ptr(0));
        rotated_lt_desc->set_dnnl_md(rotated_md);

        auto cvt_lt_n = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_m->input_value(0),
                                                                          rotated_lt_desc);
        cvt_lt_n->set_op_annotations(cvt_lt_m->get_op_annotations());

        auto reshape_n = std::make_shared<ngraph::op::v0::Reshape>(
            cvt_lt_n, reshape_order, cvt_lt_m->get_output_shape(0));
        auto reshape_n_layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(
            *reshape_n->get_output_tensor_ptr(0));
        reshape_n_layout->set_dnnl_md(out_md);
        reshape_n->get_output_tensor_ptr(0)->set_tensor_layout(reshape_n_layout);
        reshape_n->set_op_annotations(reshape_m->get_op_annotations());

        ngraph::replace_node(cvt_lt_m, reshape_n);
        NGRAPH_DEBUG << "ReshapeConvertLayout: Reordering reshape and convertlayout for faster "
                        "DNNL kernels";

        return true;
    };

    auto m = make_shared<pattern::Matcher>(
        cvt_lt, "CPUPostLayoutOptimizations.ConstructReshapeConvertLayoutFusion");
    this->add_matcher(m, callback);
}

// fold Constant + ConvertLayout to Constant
template <typename T>
static shared_ptr<ngraph::op::v0::Constant> fold_constant_convertlayout_helper(
    const shared_ptr<op::v0::Constant>& input,
    const shared_ptr<runtime::cpu::op::ConvertLayout>& convertlayout,
    dnnl::memory::desc& input_desc,
    dnnl::memory::desc& result_desc)
{
    std::vector<T> result_vec(convertlayout->get_output_tensor(0).size() /
                              input->get_output_element_type(0).size());

    bool input_format_is_nchw = runtime::cpu::dnnl_utils::dnnl_md_matches_format_tag(
        input_desc.data, dnnl::memory::format_tag::nchw);
    if (input_format_is_nchw && runtime::cpu::dnnl_utils::dnnl_md_matches_format_tag(
                                    result_desc.data, dnnl::memory::format_tag::goihw))
    {
        // becomes a copy
        input_desc = result_desc;
    }
    else if ((input_format_is_nchw || runtime::cpu::dnnl_utils::dnnl_md_matches_format_tag(
                                          input_desc.data, dnnl::memory::format_tag::nhwc)) &&
             (runtime::cpu::dnnl_utils::dnnl_md_matches_format_tag(
                  result_desc.data, dnnl::memory::format_tag::OIhw4i16o4i) &&
              // check if compensation is conv_s8s8(1U)
              result_desc.data.extra.flags & 0x1U))
    {
        auto arg0_shape = input->get_output_shape(0);
        input_desc = dnnl::memory::desc(
            dnnl::memory::dims(arg0_shape.begin(), arg0_shape.end()),
            runtime::cpu::dnnl_utils::get_dnnl_data_type(input->get_output_element_type(0)),
            dnnl::memory::format_tag::oihw);
    }
    else if (input_format_is_nchw && input_desc.data.ndims == 4 && result_desc.data.ndims == 5 &&
             convertlayout->get_users().size() == 1)
    {
        Shape weights_shape_groups;
        if (auto gconv =
                as_type_ptr<ngraph::op::v0::GroupConvolution>(convertlayout->get_users()[0]))
        {
            weights_shape_groups = gconv->get_weights_dimensions();
        }
        else if (auto gconvb =
                     as_type_ptr<ngraph::op::GroupConvolutionBias>(convertlayout->get_users()[0]))
        {
            weights_shape_groups = gconvb->get_weights_dimensions();
        }
        else
        {
            throw ngraph_error("Incompatible input/output shape in ConvertLayout op");
        }
        input_desc = dnnl::memory::desc(
            dnnl::memory::dims(weights_shape_groups.begin(), weights_shape_groups.end()),
            runtime::cpu::dnnl_utils::get_dnnl_data_type(input->get_output_element_type(0)),
            dnnl::memory::format_tag::goihw);
    }

    // build dnnl primitive and execute
    dnnl::memory in{input_desc,
                    runtime::cpu::executor::global_cpu_engine,
                    const_cast<void*>(input->get_data_ptr())};
    dnnl::memory out{result_desc, runtime::cpu::executor::global_cpu_engine, result_vec.data()};
    dnnl::reorder reorder{in, out};

    std::unordered_map<int, dnnl::memory> exec_args = {{DNNL_ARG_SRC, in}, {DNNL_ARG_DST, out}};

    dnnl::stream s(runtime::cpu::executor::global_cpu_engine);
    try
    {
        reorder.execute(s, exec_args);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    return make_shared<ngraph::op::v0::Constant>(
        convertlayout->get_output_element_type(0), convertlayout->get_output_shape(0), result_vec);
}

bool ngraph::runtime::cpu::pass::CPUConvertLayoutConstantFolding::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    auto replace = false;
    for (auto n : function->get_ordered_ops())
    {
        if (is_type<runtime::cpu::op::ConvertLayout>(n))
        {
            auto m_convertlayout = static_pointer_cast<runtime::cpu::op::ConvertLayout>(n);
            auto output_md = dnnl_utils::get_output_dnnl_md(m_convertlayout.get(), 0);
            // do not do constant folding if the output is padded data layout
            if (dnnl_utils::is_dnnl_padded_layout(
                    output_md, ngraph::get_default_order(m_convertlayout->get_output_shape(0))))
            {
                continue;
            }

            auto arg = m_convertlayout->get_input_node_shared_ptr(0);
            if (is_type<ngraph::op::v0::Constant>(arg))
            {
                auto m_input = static_pointer_cast<ngraph::op::v0::Constant>(arg);
                auto input_md = dnnl_utils::get_input_dnnl_md(m_convertlayout.get(), 0);

                std::shared_ptr<ngraph::op::v0::Constant> replacement;

                switch (m_input->get_output_element_type(0))
                {
                case element::Type_t::undefined:
                    NGRAPH_CHECK(
                        false,
                        "Encountered 'undefined' element type in construct_constant_convertlayout");
                    break;
                case element::Type_t::dynamic:
                    NGRAPH_CHECK(
                        false,
                        "Encountered 'dynamic' element type in construct_constant_convertlayout");
                    break;
                case element::Type_t::u1:
                    NGRAPH_CHECK(
                        false, "Encountered 'u1' element type in construct_constant_convertlayout");
                    break;
                case element::Type_t::boolean:
                    replacement = fold_constant_convertlayout_helper<char>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::bf16:
                    replacement = fold_constant_convertlayout_helper<bfloat16>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::f16:
                    replacement = fold_constant_convertlayout_helper<float16>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::f32:
                    replacement = fold_constant_convertlayout_helper<float>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::f64:
                    replacement = fold_constant_convertlayout_helper<double>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::i8:
                    replacement = fold_constant_convertlayout_helper<int8_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::i16:
                    replacement = fold_constant_convertlayout_helper<int16_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::i32:
                    replacement = fold_constant_convertlayout_helper<int32_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::i64:
                    replacement = fold_constant_convertlayout_helper<int64_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::u8:
                    replacement = fold_constant_convertlayout_helper<uint8_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::u16:
                    replacement = fold_constant_convertlayout_helper<uint16_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::u32:
                    replacement = fold_constant_convertlayout_helper<uint32_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                case element::Type_t::u64:
                    replacement = fold_constant_convertlayout_helper<uint64_t>(
                        m_input, m_convertlayout, input_md, output_md);
                    break;
                }

                auto tv = replacement->get_output_tensor_ptr(0);
                auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
                layout->set_dnnl_md(output_md);
                tv->set_tensor_layout(layout);
                replace_node(n, replacement);
                replace = true;
            }
        }
    }
    return replace;
}
