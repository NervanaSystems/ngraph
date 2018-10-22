/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/plaidml/plaidml_compiled_function.hpp"

#include <utility>

#include "ngraph/log.hpp"
#include "ngraph/runtime/plaidml/plaidml_build.hpp"
#include "ngraph/runtime/plaidml/plaidml_tensor.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::CompiledFunction::CompiledFunction(Build build)
    : config_{build.config}
    , func_{std::move(build.func)}
    , input_names_{std::move(build.input_names)}
    , output_names_{std::move(build.output_names)}
    , invoker_{build.config->ctx, std::move(build.composer)}
{
    NGRAPH_DEBUG << "Compiled PlaidML function " << this;
}

bool ngraph::runtime::plaidml::CompiledFunction::schedule_invocation(
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs) const
{
    std::lock_guard<std::mutex> lock{mu_};

    NGRAPH_DEBUG << "Binding PlaidML function " << this;

    std::size_t input_count = 0;
    for (const auto& param : func_->get_parameters())
    {
        for (std::size_t idx = 0; idx < param->get_output_size(); ++idx)
        {
            descriptor::Tensor* tv = param->get_output_tensor_ptr(idx).get();
            auto rtv = dynamic_cast<PlaidML_Tensor*>(inputs[input_count++].get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operations on PlaidML tensor views"};
            }
            rtv->sync_input();
            NGRAPH_DEBUG << "Binding input " << input_names_.at(tv) << " to tensor " << rtv;
            invoker_.set_input(input_names_.at(tv), rtv->tensor());
        }
    }

    std::size_t output_count = 0;
    for (const auto& result : func_->get_results())
    {
        for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
        {
            descriptor::Tensor* tv = result->get_output_tensor_ptr(idx).get();
            auto rtv = dynamic_cast<PlaidML_Tensor*>(outputs[output_count++].get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operations on PlaidML tensor views"};
            }
            NGRAPH_DEBUG << "Binding output " << output_names_.at(tv) << " to tensor " << rtv;
            invoker_.set_output(output_names_.at(tv), rtv->tensor());
        }
    }

    NGRAPH_DEBUG << "Invoking PlaidML function " << this;

    invoker_.invoke();
    bound_ = true;

    output_count = 0;
    for (const auto& result : func_->get_results())
    {
        for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
        {
            auto rtv = dynamic_cast<PlaidML_Tensor*>(outputs[output_count++].get());
            if (!rtv)
            {
                throw std::runtime_error{
                    "The PlaidML backend only operations on PlaidML tensor views"};
            }
            rtv->sync_output();
        }
    }
    return true;
}

void ngraph::runtime::plaidml::CompiledFunction::save(const std::string& filename,
                                                      plaidml_file_format format) const
{
    std::lock_guard<std::mutex> lock{mu_};

    if (!bound_)
    {
        for (const auto& param : func_->get_parameters())
        {
            for (std::size_t idx = 0; idx < param->get_output_size(); ++idx)
            {
                descriptor::Tensor* tv = param->get_output_tensor_ptr(idx).get();
                auto tensor = config_->dev->allocate(
                    to_plaidml(config_->ctx, tv->get_element_type(), tv->get_shape()));
                invoker_.set_input(input_names_.at(tv), tensor);
            }
        }
        for (const auto& result : func_->get_results())
        {
            for (std::size_t idx = 0; idx < result->get_output_size(); ++idx)
            {
                descriptor::Tensor* tv = result->get_output_tensor_ptr(idx).get();
                auto tensor = config_->dev->allocate(
                    to_plaidml(config_->ctx, tv->get_element_type(), tv->get_shape()));
                invoker_.set_output(output_names_.at(tv), tensor);
            }
        }
        bound_ = true;
    }

    invoker_.save(filename, format);
}
