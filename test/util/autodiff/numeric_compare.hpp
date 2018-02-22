/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/log.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/types/element_type.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_derivative.hpp"
#include "util/autodiff/numeric_derivative.hpp"
#include "util/test_tools.hpp"

template <typename T>
bool autodiff_numeric_compare(const std::shared_ptr<ngraph::runtime::Manager>& manager,
                              const std::shared_ptr<ngraph::runtime::Backend>& backend,
                              std::function<std::shared_ptr<ngraph::Function>()> make_graph,
                              const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args,
                              T rtol,
                              T atol)
{
    T delta = static_cast<T>(0.001);

    // Use CPU to compute numerical derivatives
    auto cpu_manager = ngraph::runtime::Manager::get("CPU");
    auto cpu_backend = cpu_manager->allocate_backend();
    auto f = make_graph();

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> args_cpu;
    for (auto arg : args)
    {
        auto arg_cpu = cpu_backend->make_primary_tensor_view(arg->get_tensor().get_element_type(),
                                                             arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_tensor().get_element_type() == ngraph::element::boolean)
        {
            copy_data(arg_cpu, read_vector<char>(arg));
        }
        else
        {
            copy_data(arg_cpu, read_vector<T>(arg));
        }
        args_cpu.push_back(arg_cpu);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        cpu_manager, cpu_backend, f, args_cpu, delta, f->get_parameters());

    // Use the backend being tested to compute symbolic derivatives
    auto g = make_graph();
    auto results_sym =
        ngraph::autodiff::backprop_derivative<T>(manager, backend, g, args, g->get_parameters());

    // Cast to HostTensorView for comparision
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> results_sym_cpu;
    for (auto result : results_sym)
    {
        auto result_cpu =
            cpu_backend->make_primary_tensor_view(ngraph::element::from<T>(), result->get_shape());
        copy_data(result_cpu, read_vector<T>(result));
        results_sym_cpu.push_back(result_cpu);
    }

    return ngraph::test::all_close(results_num, results_sym_cpu, rtol, atol);
}

template <typename T>
bool autodiff_numeric_compare_selective(
    const std::shared_ptr<ngraph::runtime::Manager>& manager,
    const std::shared_ptr<ngraph::runtime::Backend>& backend,
    std::function<std::shared_ptr<ngraph::Function>()> make_graph,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args,
    T rtol,
    T atol,
    const std::vector<bool>& indep_param_mask)
{
    // Use CPU to compute numerical derivatives
    std::vector<std::shared_ptr<ngraph::op::Parameter>> f_indep_params;
    auto f = make_graph();

    size_t i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            f_indep_params.push_back(f->get_parameters().at(i));
        }
        i++;
    }

    auto cpu_manager = ngraph::runtime::Manager::get("CPU");
    auto cpu_backend = cpu_manager->allocate_backend();

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> args_cpu;
    for (auto arg : args)
    {
        auto arg_cpu = cpu_backend->make_primary_tensor_view(arg->get_tensor().get_element_type(),
                                                             arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_tensor().get_element_type() == ngraph::element::boolean)
        {
            copy_data(arg_cpu, read_vector<char>(arg));
        }
        else
        {
            copy_data(arg_cpu, read_vector<T>(arg));
        }
        args_cpu.push_back(arg_cpu);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        cpu_manager, cpu_backend, f, args_cpu, .001f, f_indep_params);

    // Use the backend being tested to compute symbolic derivatives
    std::vector<std::shared_ptr<ngraph::op::Parameter>> g_indep_params;
    auto g = make_graph();

    i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            g_indep_params.push_back(g->get_parameters().at(i));
        }
        i++;
    }

    auto results_sym =
        ngraph::autodiff::backprop_derivative<T>(manager, backend, g, args, g_indep_params);

    // Cast to HostTensorView for comparision
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> results_sym_cpu;
    for (auto result : results_sym)
    {
        auto result_cpu =
            cpu_backend->make_primary_tensor_view(ngraph::element::from<T>(), result->get_shape());
        copy_data(result_cpu, read_vector<T>(result));
        results_sym_cpu.push_back(result_cpu);
    }

    return ngraph::test::all_close(results_num, results_sym_cpu, rtol, atol);
}
