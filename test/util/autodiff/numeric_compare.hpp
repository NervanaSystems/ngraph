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
#include "ngraph/type/element_type.hpp"
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

    // Use INTERPRETER to compute numerical derivatives
    auto interpreter_manager = ngraph::runtime::Manager::get("INTERPRETER");
    auto interpreter_backend = interpreter_manager->allocate_backend();
    auto f = make_graph();

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> interpreter_args;
    for (auto arg : args)
    {
        auto interpreter_arg = interpreter_backend->make_primary_tensor_view(
            arg->get_tensor().get_element_type(), arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_tensor().get_element_type() == ngraph::element::boolean)
        {
            copy_data(interpreter_arg, read_vector<char>(arg));
        }
        else
        {
            copy_data(interpreter_arg, read_vector<T>(arg));
        }
        interpreter_args.push_back(interpreter_arg);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        interpreter_manager, interpreter_backend, f, interpreter_args, delta, f->get_parameters());

    // Use the backend being tested to compute symbolic derivatives
    auto g = make_graph();
    auto results_sym =
        ngraph::autodiff::backprop_derivative<T>(manager, backend, g, args, g->get_parameters());

    // Cast to HostTensorView for comparision
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> interpreter_results_sym;
    for (auto result : results_sym)
    {
        auto interpreter_result = interpreter_backend->make_primary_tensor_view(
            ngraph::element::from<T>(), result->get_shape());
        copy_data(interpreter_result, read_vector<T>(result));
        interpreter_results_sym.push_back(interpreter_result);
    }

    return ngraph::test::all_close(results_num, interpreter_results_sym, rtol, atol);
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
    // Use INTERPRETER to compute numerical derivatives
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

    auto interpreter_manager = ngraph::runtime::Manager::get("INTERPRETER");
    auto interpreter_backend = interpreter_manager->allocate_backend();

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> interpreter_args;
    for (auto arg : args)
    {
        auto interpreter_arg = interpreter_backend->make_primary_tensor_view(
            arg->get_tensor().get_element_type(), arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_tensor().get_element_type() == ngraph::element::boolean)
        {
            copy_data(interpreter_arg, read_vector<char>(arg));
        }
        else
        {
            copy_data(interpreter_arg, read_vector<T>(arg));
        }
        interpreter_args.push_back(interpreter_arg);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        interpreter_manager, interpreter_backend, f, interpreter_args, .001f, f_indep_params);

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
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> interpreter_results_sym;
    for (auto result : results_sym)
    {
        auto interpreter_result = interpreter_backend->make_primary_tensor_view(
            ngraph::element::from<T>(), result->get_shape());
        copy_data(interpreter_result, read_vector<T>(result));
        interpreter_results_sym.push_back(interpreter_result);
    }

    return ngraph::test::all_close(results_num, interpreter_results_sym, rtol, atol);
}
