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

#include <ops.hpp>
#include "constant_folding.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

void next_coordinates(vector<int64_t>& coordinates, const Shape& max_shape)
{
    for (int64_t i = coordinates.size() - 1; i >= 0; --i)
    {
        if (coordinates[i] < max_shape[i] - 1)
        {
            ++coordinates[i];
            return;
        }
        else
        {
            coordinates[i] = 0;
        }
    }
}

template <typename T>
static shared_ptr<op::Constant>
    fold_constant_non_zero_helper(const shared_ptr<op::Constant>& input_constant, const T& zero)
{
    auto shape = input_constant->get_shape();
    auto rank = shape.size();
    auto input_vector = input_constant->template cast_vector<T>();
    vector<vector<int64_t>> result(rank, vector<int64_t>());

    vector<int64_t> curr_coordinates(rank, 0);
    for (const auto& value : input_vector)
    {
        if (value != zero)
        {
            for (auto i = 0; i < rank; ++i)
                result[i].push_back(curr_coordinates[i]);
        }
        next_coordinates(curr_coordinates, shape);
    }
    auto flattened_result = vector<int64_t>();
    for (const auto& row : result)
        flattened_result.insert(flattened_result.end(), row.begin(), row.end());

    return make_shared<op::Constant>(
        element::i64, Shape{result.size(), result[0].size()}, flattened_result);
}

void pass::ConstantFolding::construct_constant_non_zero()
{
    auto const_op = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto non_zero_op = make_shared<op::v2::NonZero>(const_op);

    auto constant_non_zero_callback = [const_op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_non_zero_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto constant_node = static_pointer_cast<op::Constant>(pattern_map[const_op]);
        auto non_zero_node = static_pointer_cast<op::v2::NonZero>(m.get_match_root());

        std::shared_ptr<op::Constant> replacement;

        switch (non_zero_node->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_non_zero");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_non_zero");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_non_zero");
            break;
        case element::Type_t::boolean:
            NGRAPH_CHECK(false, "Encountered 'boolean' element type in fold_constant_non_zero");
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_non_zero_helper<bfloat16>(constant_node, bfloat16());
            break;
        case element::Type_t::f16:
            replacement = fold_constant_non_zero_helper<float16>(constant_node, float16());
            break;
        case element::Type_t::f32:
            replacement = fold_constant_non_zero_helper<float>(constant_node, float(0));
            break;
        case element::Type_t::f64:
            replacement = fold_constant_non_zero_helper<double>(constant_node, double(0));
            break;
        case element::Type_t::i8:
            replacement = fold_constant_non_zero_helper<int8_t>(constant_node, int8_t(0));
            break;
        case element::Type_t::i16:
            replacement = fold_constant_non_zero_helper<int16_t>(constant_node, int16_t(0));
            break;
        case element::Type_t::i32:
            replacement = fold_constant_non_zero_helper<int32_t>(constant_node, int32_t(0));
            break;
        case element::Type_t::i64:
            replacement = fold_constant_non_zero_helper<int64_t>(constant_node, int64_t(0));
            break;
        case element::Type_t::u8:
            replacement = fold_constant_non_zero_helper<uint8_t>(constant_node, uint8_t(0));
            break;
        case element::Type_t::u16:
            replacement = fold_constant_non_zero_helper<uint16_t>(constant_node, uint16_t(0));
            break;
        case element::Type_t::u32:
            replacement = fold_constant_non_zero_helper<uint32_t>(constant_node, uint32_t(0));
            break;
        case element::Type_t::u64:
            replacement = fold_constant_non_zero_helper<uint64_t>(constant_node, uint64_t(0));
            break;
        }
        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto non_zero_matcher =
        make_shared<pattern::Matcher>(non_zero_op, "ConstantFolding.ConstantNonZero");
    this->add_matcher(
        non_zero_matcher, constant_non_zero_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
