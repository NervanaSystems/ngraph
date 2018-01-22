// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "util/all_close.hpp"
#include "util/autodiff/backprop_derivative.hpp"
#include "util/autodiff/numeric_derivative.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
bool autodiff_numeric_compare(const shared_ptr<runtime::Manager>& manager,
                              const shared_ptr<runtime::Backend>& backend,
                              function<shared_ptr<Function>()> make_graph,
                              const vector<shared_ptr<runtime::TensorView>>& args,
                              T rtol,
                              T atol)
{
    auto f = make_graph();
    auto results_num =
        autodiff::numeric_derivative<T>(manager, backend, f, args, .001f, f->get_parameters());

    auto g = make_graph();
    auto results_sym =
        autodiff::backprop_derivative<T>(manager, backend, g, args, g->get_parameters());

    return test::all_close(results_num, results_sym, rtol, atol);
}

template <typename T>
bool autodiff_numeric_compare_selective(const shared_ptr<runtime::Manager>& manager,
                                        const shared_ptr<runtime::Backend>& backend,
                                        function<shared_ptr<Function>()> make_graph,
                                        const vector<shared_ptr<runtime::TensorView>>& args,
                                        T rtol,
                                        T atol,
                                        const vector<bool>& indep_param_mask)
{
    vector<shared_ptr<op::Parameter>> f_indep_params;
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
        autodiff::numeric_derivative<T>(manager, backend, f, args, .001f, f_indep_params);

    vector<shared_ptr<op::Parameter>> g_indep_params;
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

    auto results_sym = autodiff::backprop_derivative<T>(manager, backend, g, args, g_indep_params);

    return test::all_close(results_num, results_sym, rtol, atol);
}
