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

#include "test_case.hpp"
#include "all_close.hpp"
#include "all_close_f.hpp"
#include "gtest/gtest.h"
#include "ngraph/assertion.hpp"
#include "test_tools.hpp"

namespace
{
    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value, ::testing::AssertionResult>::type
        compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                       const std::shared_ptr<ngraph::runtime::Tensor>& results)
    {
        const auto expected = expected_results->get_vector<T>();
        const auto result = read_vector<T>(results);
        return ngraph::test::all_close_f(expected, result);
    }

    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type
        compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                       const std::shared_ptr<ngraph::runtime::Tensor>& results)
    {
        const auto expected = expected_results->get_vector<T>();
        const auto result = read_vector<T>(results);
        return ngraph::test::all_close(expected, result);
    }

    using value_comparator_function =
        std::function<::testing::AssertionResult(const std::shared_ptr<ngraph::op::Constant>&,
                                                 const std::shared_ptr<ngraph::runtime::Tensor>&)>;

    const std::map<ngraph::element::Type_t, value_comparator_function> value_comparators = {
        {ngraph::element::Type_t::f32, compare_values<float>},
        {ngraph::element::Type_t::f64, compare_values<double>},
        {ngraph::element::Type_t::i8, compare_values<int8_t>},
        {ngraph::element::Type_t::i16, compare_values<int16_t>},
        {ngraph::element::Type_t::i32, compare_values<int32_t>},
        {ngraph::element::Type_t::i64, compare_values<int64_t>},
        {ngraph::element::Type_t::u8, compare_values<uint8_t>},
        {ngraph::element::Type_t::u16, compare_values<uint16_t>},
        {ngraph::element::Type_t::u32, compare_values<uint32_t>},
        {ngraph::element::Type_t::u64, compare_values<uint64_t>}};
}

void ngraph::test::NgraphTestCase::run()
{
    const auto& function_results = m_function->get_results();
    NGRAPH_CHECK(m_expected_outputs.size() == function_results.size(),
                 "Expected number of outputs is different from the function's number of results.");

    auto handle = m_backend->compile(m_function);
    handle->call_with_validate(m_result_tensors, m_input_tensors);

    for (int i = 0; i < m_expected_outputs.size(); ++i)
    {
        const auto& result_tensor = m_result_tensors.at(i);
        const auto& expected_result_constant = m_expected_outputs.at(i);
        const auto& element_type = result_tensor->get_element_type();

        if (value_comparators.count(element_type.get_type_enum()) == 0)
        {
            NGRAPH_FAIL() << "Please add support for " << element_type
                          << " to ngraph::test::NgraphTestCase::run()";
        }
        else
        {
            auto values_match = value_comparators.at(element_type.get_type_enum());

            EXPECT_TRUE(values_match(expected_result_constant, result_tensor));
        }
    }
}
