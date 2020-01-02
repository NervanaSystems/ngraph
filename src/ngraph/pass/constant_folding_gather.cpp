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

#include "constant_folding.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/runtime/reference/gather.hpp"

using namespace std;
using namespace ngraph;

// "Inner" helper for fold_constant_gather, which has to switch on the indices
// element type.
template <typename T, typename U>
static shared_ptr<op::Constant> fold_constant_gather_helper(const shared_ptr<op::Constant>& data,
                                                            const shared_ptr<op::Constant>& indices,
                                                            const shared_ptr<Node>& gather)
{
    runtime::AlignedBuffer buffer(shape_size(gather->get_shape()) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    if (auto gather_v1 = as_type_ptr<op::v1::Gather>(gather))
    {
        runtime::reference::gather<T, U>(data->get_data_ptr<T>(),
                                         indices->get_data_ptr<U>(),
                                         data_ptr,
                                         data->get_shape(),
                                         indices->get_shape(),
                                         gather_v1->get_shape(),
                                         gather_v1->get_axis());
    }
    else if (auto gather_v0 = as_type_ptr<op::v0::Gather>(gather))
    {
        runtime::reference::gather<T, U>(data->get_data_ptr<T>(),
                                         indices->get_data_ptr<U>(),
                                         data_ptr,
                                         data->get_shape(),
                                         indices->get_shape(),
                                         gather_v0->get_shape(),
                                         gather_v0->get_axis());
    }
    else
    {
        throw ngraph_error("Unsupported op in gather constant folding.");
    }

    return make_shared<op::Constant>(
        gather->get_output_element_type(0), gather->get_output_shape(0), data_ptr);
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_gather(const shared_ptr<op::Constant>& data,
                                                     const shared_ptr<op::Constant>& indices,
                                                     const shared_ptr<Node>& gather)
{
    auto indices_type = indices->get_output_element_type(0);

    switch (indices_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_gather_callback");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_gather_callback");
        break;
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::u1:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
        NGRAPH_CHECK(false,
                     "Encountered unsupported indices element type in constant_gather_callback: ",
                     indices_type);
        break;
    case element::Type_t::i32:
        return fold_constant_gather_helper<T, int32_t>(data, indices, gather);
    case element::Type_t::i64:
        return fold_constant_gather_helper<T, int64_t>(data, indices, gather);
    }

    NGRAPH_UNREACHABLE("Unhandled switch case");
}

void pass::ConstantFolding::construct_constant_gather()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{10, 20, 30}, pattern::has_class<op::Constant>());
    auto indices_label =
        make_shared<pattern::op::Label>(element::i64, Shape{5}, pattern::has_class<op::Constant>());
    size_t gather_axis = 1;
    auto gather_v0 = make_shared<op::v0::Gather>(data_label, indices_label, gather_axis);

    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());
    auto gather_v1 = make_shared<op::v1::Gather>(data_label, indices_label, axis_label);

    auto constant_gather_callback = [data_label, indices_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_gather_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto indices = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        auto gather = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(gather));

        std::shared_ptr<Node> replacement;
        auto data_type = data->get_output_element_type(0);
        auto indices_type = indices->get_output_element_type(0);
        switch (data_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_gather_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_gather_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_gather_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_gather<char>(data, indices, gather);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_gather<bfloat16>(data, indices, gather);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_gather<float16>(data, indices, gather);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_gather<float>(data, indices, gather);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_gather<double>(data, indices, gather);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_gather<int8_t>(data, indices, gather);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_gather<int16_t>(data, indices, gather);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_gather<int32_t>(data, indices, gather);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_gather<int64_t>(data, indices, gather);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_gather<uint8_t>(data, indices, gather);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_gather<uint16_t>(data, indices, gather);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_gather<uint32_t>(data, indices, gather);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_gather<uint64_t>(data, indices, gather);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto gather_matcher_v0 =
        make_shared<pattern::Matcher>(gather_v0, "ConstantFolding.ConstantGatherV0");
    this->add_matcher(
        gather_matcher_v0, constant_gather_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    auto gather_matcher_v1 =
        make_shared<pattern::Matcher>(gather_v1, "ConstantFolding.ConstantGatherV1");
    this->add_matcher(
        gather_matcher_v1, constant_gather_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
