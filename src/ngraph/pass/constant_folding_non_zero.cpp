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

#include <numeric>

#include "constant_folding.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/runtime/reference/non_zero.hpp"

using namespace std;
using namespace ngraph;

template <typename T, typename U>
static shared_ptr<op::Constant>
    fold_constant_non_zero_execute(const shared_ptr<op::Constant>& data,
                                   const element::Type& index_element_type)
{
    const auto input_shape = data->get_shape();
    size_t input_rank = input_shape.size();
    const auto* input_values = data->get_data_ptr<T>();
    Shape out_shape;

    size_t non_zero_count = runtime::reference::non_zero_get_count<T>(input_values, input_shape);
    size_t out_elem_count = (input_rank == 0) ? non_zero_count : (input_rank * non_zero_count);

    if (out_elem_count == 0)
    {
        out_shape = Shape{0};
    }
    else if (input_rank == 0)
    {
        out_shape = Shape{1, 1};
    }
    else
    {
        out_shape = Shape{input_rank, non_zero_count};
    }

#if 1
    U* data_ptr = nullptr;
    size_t out_data_byte_size = (index_element_type == element::i64)
                                    ? (out_elem_count * sizeof(int64_t))
                                    : (out_elem_count * sizeof(int32_t));

    if (out_elem_count > 0)
    {
        // runtime::AlignedBuffer buffer(out_data_byte_size);
        // data_ptr = buffer.get_ptr<U>();
        data_ptr = reinterpret_cast<U*>(malloc(out_data_byte_size));
    }

    runtime::reference::non_zero<T, U>(input_values, data_ptr, input_shape);
    return make_shared<op::Constant>(index_element_type, out_shape, data_ptr);
#else
    if (index_element_type == element::i64)
    {
        runtime::AlignedBuffer buffer(out_elem_count * sizeof(int64_t));
        int64_t* data_ptr = buffer.get_ptr<int64_t>();
        runtime::reference::non_zero<T, int64_t>(input_values, data_ptr, input_shape);
        return make_shared<op::Constant>(index_element_type, out_shape, data_ptr);
    }
    else
    {
        runtime::AlignedBuffer buffer(out_elem_count * sizeof(int32_t));
        int32_t* data_ptr = buffer.get_ptr<int32_t>();
        runtime::reference::non_zero<T, int32_t>(input_values, data_ptr, input_shape);
        return make_shared<op::Constant>(index_element_type, out_shape, data_ptr);
    }
#endif
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_non_zero(const shared_ptr<op::Constant>& data,
                                                       const element::Type& index_element_type)
{
    if (index_element_type == element::i64)
    {
        return fold_constant_non_zero_execute<T, int64_t>(data, index_element_type);
    }
    else if (index_element_type == element::i32)
    {
        return fold_constant_non_zero_execute<T, int32_t>(data, index_element_type);
    }
    else
    {
        NGRAPH_CHECK(false, "Only i32 or i64 type is accepted for NonZero output");
        return nullptr;
    }
}

void pass::ConstantFolding::construct_constant_non_zero()
{
    const auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 2, 3}, pattern::has_class<op::Constant>());
    const auto non_zero = make_shared<op::v3::NonZero>(data_label);

    auto constant_non_zero_callback = [data_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        const auto data = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        const auto non_zero_matched = as_type_ptr<op::v3::NonZero>(m.get_match_root());
        auto output_type = non_zero_matched->get_output_type();

        std::shared_ptr<Node> replacement;
        switch (data->get_element_type())
        {
        case element::Type_t::boolean:
            replacement = fold_constant_non_zero<char>(data, output_type);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_non_zero<bfloat16>(data, output_type);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_non_zero<float16>(data, output_type);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_non_zero<float>(data, output_type);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_non_zero<double>(data, output_type);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_non_zero<int8_t>(data, output_type);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_non_zero<int16_t>(data, output_type);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_non_zero<int32_t>(data, output_type);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_non_zero<int64_t>(data, output_type);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_non_zero<uint8_t>(data, output_type);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_non_zero<uint16_t>(data, output_type);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_non_zero<uint32_t>(data, output_type);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_non_zero<uint64_t>(data, output_type);
            break;
        case element::Type_t::u1:
        case element::Type_t::dynamic:
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Unsupported data type in NonZero constant folding");
            break;
        }

        if (replacement.get() != nullptr)
        {
            replace_node(m.get_match_root(), replacement);
            return true;
        }
        else
        {
            return false;
        }
    };

    const auto matcher =
        make_shared<pattern::Matcher>(non_zero, "ConstantFolding.ConstantNonZeroV3");
    this->add_matcher(matcher, constant_non_zero_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
