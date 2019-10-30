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

#pragma once

#include <utility>

#include "all_close.hpp"
#include "all_close_f.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        class NgraphTestCase
        {
        public:
            NgraphTestCase(const std::shared_ptr<Function>& function,
                           const std::string& backend_name)
                : m_function(function)
                , m_backend(ngraph::runtime::Backend::create(backend_name))
            {
            }

            /// \brief Makes the test case print the expected and computed values to the console.
            ///        This should only be used for debugging purposes.
            ///
            /// Just before the assertion is done, the current test case will gather expected and
            /// computed values, format them as 2 columns and print out to the console along with
            //  a corresponding index in the vector.
            ///
            /// \param dump - Indicates if the test case should perform the console printout
            NgraphTestCase& dump_results(bool dump = true);

            template <typename T>
            void add_input(const std::vector<T>& values)
            {
                auto params = m_function->get_parameters();

                NGRAPH_CHECK(m_input_index < params.size(),
                             "All function parameters already have inputs.");

                auto tensor = m_backend->create_tensor(params.at(m_input_index)->get_element_type(),
                                                       params.at(m_input_index)->get_shape());
                copy_data(tensor, values);

                m_input_tensors.push_back(tensor);

                ++m_input_index;
            }

            template <typename T>
            void add_input_from_file(const std::string& basepath, const std::string& filename)
            {
                auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_input_from_file<T>(filepath);
            }

            template <typename T>
            void add_input_from_file(const std::string& filepath)
            {
                auto value = read_binary_file<T>(filepath);
                add_input(value);
            }

            template <typename T>
            void add_multiple_inputs(const std::vector<std::vector<T>>& vector_of_values)
            {
                for (const auto& value : vector_of_values)
                {
                    add_input(value);
                }
            }

            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
                auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                auto function_output_type = results.at(m_output_index)->get_element_type();
                m_result_tensors.emplace_back(
                    m_backend->create_tensor(function_output_type, expected_shape));

                m_expected_outputs.emplace_back(std::make_shared<ngraph::op::Constant>(
                    function_output_type, expected_shape, values));

                ++m_output_index;
            }

            template <typename T>
            void add_expected_output(const std::vector<T>& values)
            {
                auto shape = m_function->get_results().at(m_output_index)->get_shape();
                add_expected_output(shape, values);
            }

            template <typename T>
            void add_expected_output_from_file(ngraph::Shape expected_shape,
                                               const std::string& basepath,
                                               const std::string& filename)
            {
                auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_expected_output_from_file<T>(expected_shape, filepath);
            }

            template <typename T>
            void add_expected_output_from_file(ngraph::Shape expected_shape,
                                               const std::string& filepath)
            {
                auto value = read_binary_file<T>(filepath);
                add_expected_output(expected_shape, value);
            }

            void run(size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS);

        private:
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value,
                                    ::testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                if (m_dump_results)
                {
                    std::cout << get_results_str<T>(expected, result, expected.size());
                }

                return ngraph::test::all_close_f(expected, result, m_tolerance_bits);
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                if (m_dump_results)
                {
                    std::cout << get_results_str<T>(expected, result, expected.size());
                }

                return ngraph::test::all_close(expected, result);
            }

            using value_comparator_function = std::function<::testing::AssertionResult(
                const std::shared_ptr<ngraph::op::Constant>&,
                const std::shared_ptr<ngraph::runtime::Tensor>&)>;

#define REGISTER_COMPARATOR(element_type_, type_)                                                  \
    {                                                                                              \
        ngraph::element::Type_t::element_type_, std::bind(&NgraphTestCase::compare_values<type_>,  \
                                                          this,                                    \
                                                          std::placeholders::_1,                   \
                                                          std::placeholders::_2)                   \
    }

            std::map<ngraph::element::Type_t, value_comparator_function> m_value_comparators = {
                REGISTER_COMPARATOR(f32, float),
                REGISTER_COMPARATOR(f64, double),
                REGISTER_COMPARATOR(i8, int8_t),
                REGISTER_COMPARATOR(i16, int16_t),
                REGISTER_COMPARATOR(i32, int32_t),
                REGISTER_COMPARATOR(i64, int64_t),
                REGISTER_COMPARATOR(u8, uint8_t),
                REGISTER_COMPARATOR(u16, uint16_t),
                REGISTER_COMPARATOR(u32, uint32_t),
                REGISTER_COMPARATOR(u64, uint64_t),
            };
#undef REGISTER_COMPARATOR

        protected:
            std::shared_ptr<Function> m_function;
            std::shared_ptr<runtime::Backend> m_backend;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_input_tensors;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_result_tensors;
            std::vector<std::shared_ptr<ngraph::op::Constant>> m_expected_outputs;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
            bool m_dump_results = false;
            int m_tolerance_bits = DEFAULT_DOUBLE_TOLERANCE_BITS;
        };
    }
}
