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

#include <utility>

#include "ngraph/log.hpp"
#include "ngraph/runtime/plaidml/plaidml_build.hpp"
#include "ngraph/runtime/plaidml/plaidml_executable.hpp"
#include "ngraph/runtime/plaidml/plaidml_tensor.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::PlaidML_Executable::PlaidML_Executable(Build build,
                                                                 std::shared_ptr<Function> func)
    : m_config{build.config}
    , m_func{std::move(build.func)}
    , m_src_func{std::move(func)}
    , m_input_names{std::move(build.input_names)}
    , m_output_names{std::move(build.output_names)}
    , m_invoker{build.config->ctx, std::move(build.composer)}
{
    set_parameters_and_results(*m_func);
    NGRAPH_DEBUG << "Compiled PlaidML function " << this;
}

bool ngraph::runtime::plaidml::PlaidML_Executable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    std::lock_guard<std::mutex> lock{m_mu};

    NGRAPH_DEBUG << "Binding PlaidML function " << this;

    m_bound_inputs.resize(inputs.size());
    m_bound_outputs.resize(outputs.size());

    std::size_t input_count = 0;
    for (const auto& param : m_func->get_parameters())
    {
        for (std::size_t idx = 0; idx < param->get_output_size(); ++idx)
        {
            descriptor::Tensor* tv = param->get_output_tensor_ptr(idx).get();
            auto& input = inputs.at(input_count);
            auto rtv = dynamic_cast<PlaidML_Tensor*>(input.get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operates on PlaidML tensor views"};
            }
            rtv->sync_input();
            auto& bound_input = m_bound_inputs.at(input_count);
            ++input_count;
            if (bound_input.lock() == input)
            {
                // No need to re-bind this input.
                continue;
            }
            bound_input = input;
            NGRAPH_DEBUG << "Binding input " << m_input_names.at(tv) << " to tensor " << rtv;
            m_invoker.set_input(m_input_names.at(tv), rtv->tensor());
        }
    }

    std::size_t output_count = 0;
    for (const auto& result : m_func->get_results())
    {
        for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
        {
            descriptor::Tensor* tv = result->get_output_tensor_ptr(idx).get();
            auto& output = outputs.at(output_count);
            auto rtv = dynamic_cast<PlaidML_Tensor*>(output.get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operates on PlaidML tensor views"};
            }
            auto& bound_output = m_bound_outputs.at(output_count);
            ++output_count;
            if (bound_output.lock() == output)
            {
                // No need to re-bind this output.
                continue;
            }
            bound_output = output;
            NGRAPH_DEBUG << "Binding output " << m_output_names.at(tv) << " to tensor " << rtv;
            m_invoker.set_output(m_output_names.at(tv), rtv->tensor());
        }
    }

    NGRAPH_DEBUG << "Invoking PlaidML function " << this;

    m_invoker.invoke();
    m_bound = true;

    output_count = 0;
    for (const auto& result : m_func->get_results())
    {
        for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
        {
            auto rtv = dynamic_cast<PlaidML_Tensor*>(outputs[output_count++].get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operates on PlaidML tensor views"};
            }
            rtv->sync_output();
        }
    }
    return true;
}

std::vector<ngraph::runtime::PerformanceCounter>
    ngraph::runtime::plaidml::PlaidML_Executable::get_performance_data() const
{
    return std::vector<ngraph::runtime::PerformanceCounter>{};
}

void ngraph::runtime::plaidml::PlaidML_Executable::save_as_format(const std::string& filename,
                                                                  plaidml_file_format format) const
{
    std::lock_guard<std::mutex> lock{m_mu};

    if (!m_bound)
    {
        for (const auto& param : m_func->get_parameters())
        {
            for (std::size_t idx = 0; idx < param->get_output_size(); ++idx)
            {
                descriptor::Tensor* tv = param->get_output_tensor_ptr(idx).get();
                auto tensor = m_config->dev->allocate(
                    to_plaidml(m_config->ctx, tv->get_element_type(), tv->get_shape()));
                m_invoker.set_input(m_input_names.at(tv), tensor);
            }
        }
        for (const auto& result : m_func->get_results())
        {
            for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
            {
                descriptor::Tensor* tv = result->get_output_tensor_ptr(idx).get();
                auto tensor = m_config->dev->allocate(
                    to_plaidml(m_config->ctx, tv->get_element_type(), tv->get_shape()));
                m_invoker.set_output(m_output_names.at(tv), tensor);
            }
        }
        m_bound = true;
    }

    m_invoker.save(filename, format);
}
