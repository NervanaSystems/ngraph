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
            NGRAPH_CHECK(m_input_shape.size() > 0,
                         "Can't use the NonZeroElements class with a scalar shape");
        }

        template <typename T>
        void find_indices(const T* values)
        {
            m_current_index = Shape(m_input_shape.size(), 0UL);
            const auto values_count = shape_size(m_input_shape);

            const T zero_value = T{0};
            for (size_t i = 0; i + 1 < values_count; ++i)
            {
                if (values[i] != zero_value)
                {
                    add_to_results(m_current_index);
                }

                next_index();
            }

            // check the last element in the input values
            if (values_count != 0 && values[values_count - 1] != zero_value)
            {
                add_to_results(m_current_index);
            }
        }

        void generate_all_indices()
        {
            m_current_index = Shape(m_input_shape.size(), 0UL);
            size_t i = 0;
            const auto values_count = shape_size(m_input_shape);
            while (i + 1 < values_count)
            {
                add_to_results(m_current_index);
                next_index();
                ++i;
            }
            add_to_results(m_current_index);
        }

        const Results_t& get_indices() const { return m_results; }
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
        inline void next_index()
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
    const bool identical_elems_in_data = data->get_all_data_elements_bitwise_identical();

    if (identical_elems_in_data && input_values[0] == T{0})
    {
        return nullptr;
    }

    if (ngraph::is_scalar(input_shape))
    {
        return op::Constant::create(element::i64, Shape{1, 1}, {0});
    }
    else if (is_vector(input_shape))
    {
        const auto input_values_count = shape_size(input_shape);
        std::vector<int64_t> indices;
        indices.reserve(input_values_count);

        if (identical_elems_in_data)
        {
            // return a complete set of indices since all of them are non-zero
            indices.resize(input_values_count);
            std::iota(indices.begin(), indices.end(), 0);
        }
        else
        {
            const T zero_value = T{0};
            for (size_t i = 0; i < input_values_count; ++i)
            {
                if (input_values[i] != zero_value)
                {
                    indices.push_back(i);
                }
            }

            indices.shrink_to_fit();
        }

        return op::Constant::create(element::i64, Shape{1, indices.size()}, indices);
    }
    else
    {
        NonZeroElements non_zero_elems{input_shape};

        if (identical_elems_in_data)
        {
            non_zero_elems.generate_all_indices();
        }
        else
        {
            non_zero_elems.find_indices(input_values);
        }

        const auto& found_indices = non_zero_elems.get_indices();

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

        std::shared_ptr<Node> replacement;
        switch (data->get_element_type())
        {
        case element::Type_t::boolean: replacement = fold_constant_non_zero<char>(data); break;
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
