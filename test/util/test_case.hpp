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

#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"

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
            void add_multiple_inputs(const std::vector<std::vector<T>>& vector_of_values)
            {
                for (const auto& value : vector_of_values)
                {
                    add_input(value);
                }
            }

            template <typename T>
            void add_expected_output(const std::vector<T>& values)
            {
                auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                auto function_output_type = results.at(m_output_index)->get_element_type();
                auto function_output_shape = results.at(m_output_index)->get_shape();
                m_result_tensors.emplace_back(
                    m_backend->create_tensor(function_output_type, function_output_shape));

                m_expected_outputs.emplace_back(std::make_shared<ngraph::op::Constant>(
                    function_output_type, function_output_shape, values));

                ++m_output_index;
            }

            void run();

        protected:
            std::shared_ptr<Function> m_function;
            std::unique_ptr<runtime::Backend> m_backend;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_input_tensors;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_result_tensors;
            std::vector<std::shared_ptr<ngraph::op::Constant>> m_expected_outputs;
            int m_input_index = 0;
            int m_output_index = 0;
        };
    }
}
