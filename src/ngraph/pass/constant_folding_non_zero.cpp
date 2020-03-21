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
#include "ngraph/op/non_zero.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    struct NonZeroElements
    {
        using Results_t = std::vector<std::vector<int64_t>>;

        NonZeroElements(const Shape& input_shape)
            : m_input_shape{input_shape}
            , m_results{Results_t(input_shape.size())}
        {
        }

        template <typename T>
        const Results_t& find_indices(const T* values)
        {
            m_current_index = Shape(m_input_shape.size(), 0UL);
            const auto values_count = shape_size(m_input_shape);

            const T zero_value = T{0};
            for (size_t i = 0; i < values_count; ++i)
            {
                if (values[i] != zero_value)
                {
                    add_to_results(m_current_index);
                }

                // don't generate the next index when the last value has been processed
                if (i < values_count - 1)
                {
                    next_index();
                }
            }

            return m_results;
        }

    private:
        /// \brief Adds single dimensions of an index into the matching element of the results
        void add_to_results(const Shape& index)
        {
            for (size_t dim = 0; dim < index.size(); ++dim)
            {
                m_results.at(dim).push_back(index[dim]);
            }
        }

        // Generates an index pointing to the next element in the flattened tensor
        // It behaves similar to flipping bits when incrementing a binary number
        void next_index()
        {
            for (size_t dim = m_current_index.size() - 1; dim >= 0; --dim)
            {
                auto& dim_value = m_current_index.at(dim);
                if (dim_value + 1 == m_input_shape[dim])
                {
                    dim_value = 0;
                }
                else
                {
                    ++dim_value;
                    return;
                }
            }
        }

    private:
        const Shape m_input_shape;
        Results_t m_results;
        Shape m_current_index;
    };
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_non_zero(const shared_ptr<op::Constant>& data)
{
    const auto input_shape = data->get_shape();
    const auto* input_values = data->get_data_ptr<T>();
    if (ngraph::is_scalar(input_shape))
    {
        const auto scalar_value = input_values[0];

        NGRAPH_CHECK(scalar_value != T{0},
                     "It's not possible to constant fold a NonZero op for a scalar equal to zero.");

        // return 0(the only index) if the data input contains a scalar different than zero
        return op::Constant::create(element::i64, Shape{1, 1}, {0});
    }
    else if (is_vector(input_shape))
    {
        const auto input_values_count = shape_size(input_shape);
        std::vector<int64_t> indices;
        indices.reserve(input_values_count);

        const T zero_value = T{0};
        for (size_t i = 0; i < input_values_count; ++i)
        {
            if (input_values[i] != zero_value)
            {
                indices.push_back(i);
            }
        }

        indices.shrink_to_fit();
        return op::Constant::create(element::i64, Shape{1, indices.size()}, indices);
    }
    else
    {
        NonZeroElements non_zero_elems{input_shape};
        const auto& found_indices = non_zero_elems.find_indices(input_values);

        // we can't create an empty Constant indicating that no non-zero elems were found
        NGRAPH_CHECK(
            found_indices.front().size() > 0,
            "It's not possible to constant fold a NonZero op with an input containing only zeros.");

        // flatten the results and return them as a Constant
        std::vector<int64_t> indices;
        indices.reserve(found_indices.size() * found_indices.front().size());
        for (const auto& row : found_indices)
        {
            indices.insert(indices.end(), row.begin(), row.end());
        }

        const Shape out_shape{found_indices.size(), found_indices.front().size()};
        return op::Constant::create(element::i64, out_shape, indices);
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

        // const auto found_nz_node = m.get_match_root();
        // this fails because the output of NonZero is still dynamic - is it ok to remove this line?
        // or should the output shape of this op be calculated as the maximum possible number
        // of non-zero indices it can return if the input is a constant?
        // NGRAPH_CHECK(revalidate_and_ensure_static(found_nz_node));

        std::shared_ptr<Node> replacement;
        switch (data->get_element_type())
        {
        case element::Type_t::bf16: replacement = fold_constant_non_zero<bfloat16>(data); break;
        case element::Type_t::f16: replacement = fold_constant_non_zero<float16>(data); break;
        case element::Type_t::f32: replacement = fold_constant_non_zero<float>(data); break;
        case element::Type_t::f64: replacement = fold_constant_non_zero<double>(data); break;
        case element::Type_t::i8: replacement = fold_constant_non_zero<int8_t>(data); break;
        case element::Type_t::i16: replacement = fold_constant_non_zero<int16_t>(data); break;
        case element::Type_t::i32: replacement = fold_constant_non_zero<int32_t>(data); break;
        case element::Type_t::i64: replacement = fold_constant_non_zero<int64_t>(data); break;
        case element::Type_t::u8: replacement = fold_constant_non_zero<uint8_t>(data); break;
        case element::Type_t::u16: replacement = fold_constant_non_zero<uint16_t>(data); break;
        case element::Type_t::u32: replacement = fold_constant_non_zero<uint32_t>(data); break;
        case element::Type_t::u64: replacement = fold_constant_non_zero<uint64_t>(data); break;
        case element::Type_t::u1:
        case element::Type_t::boolean:
        case element::Type_t::dynamic:
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Unsupported data type in NonZero constant folding");
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    const auto matcher =
        make_shared<pattern::Matcher>(non_zero, "ConstantFolding.ConstantNonZeroV3");
    this->add_matcher(matcher, constant_non_zero_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
