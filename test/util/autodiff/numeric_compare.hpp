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

#include "util/all_close.hpp"
#include "util/autodiff/backprop_derivative.hpp"
#include "util/autodiff/numeric_derivative.hpp"

template <typename T>
bool autodiff_numeric_compare(const std::shared_ptr<ngraph::runtime::Manager>& manager,
                              const std::shared_ptr<ngraph::runtime::Backend>& backend,
                              std::function<std::shared_ptr<ngraph::Function>()> make_graph,
                              const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args,
                              T rtol,
                              T atol)
{
    T delta = static_cast<T>(0.001);
    auto f = make_graph();
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        manager, backend, f, args, delta, f->get_parameters());

    auto g = make_graph();
    auto results_sym =
        ngraph::autodiff::backprop_derivative<T>(manager, backend, g, args, g->get_parameters());

    return ngraph::test::all_close(results_num, results_sym, rtol, atol);
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

    auto results_num =
        ngraph::autodiff::numeric_derivative<T>(manager, backend, f, args, .001f, f_indep_params);

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

    return ngraph::test::all_close(results_num, results_sym, rtol, atol);
}
