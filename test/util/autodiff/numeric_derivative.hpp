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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tuple.hpp"
#include "ngraph/runtime/value.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace autodiff
    {
        /// @brief numeric approximation of the derivative
        /// @param f A function
        /// @param args Values for the arguments (the independent variables)
        /// @param delta increment for the variables
        /// @param indep_params parameters with respect to which to compute derivatives
        /// @returns vector of dy/dvar, where each dy/dvar's shape is concat(y.shape(), var.shape())
        template <typename T>
        std::vector<std::shared_ptr<runtime::TensorView>>
            numeric_derivative(const std::shared_ptr<runtime::Manager>& manager,
                               const std::shared_ptr<runtime::Backend>& backend,
                               const std::shared_ptr<Function>& f,
                               const std::vector<std::shared_ptr<runtime::TensorView>>& args,
                               T delta,
                               const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            auto y = f->get_result();

            Shape y_shape =
                std::dynamic_pointer_cast<const TensorViewType>(y->get_value_type())->get_shape();

            auto params = f->get_parameters();

            // Results for each derivative, shape Y|X_i
            std::vector<std::shared_ptr<runtime::TensorView>> results;

            for (auto param : indep_params)
            {
                Shape s = y_shape;
                auto param_shape =
                    std::dynamic_pointer_cast<const TensorViewType>(param->get_value_type())
                        ->get_shape();
                s.insert(s.end(), param_shape.begin(), param_shape.end());
                results.push_back(backend->make_primary_tensor_view<T>(s));
            }

            auto external = manager->compile(f);
            auto cf = backend->make_call_frame(external);

            // ref_y is the function evaluated at the args
            auto ref_y = backend->make_primary_tensor_view<T>(y_shape);

            cf->tensor_call(args, std::vector<std::shared_ptr<ngraph::runtime::TensorView>>{ref_y});
            auto ref_vec = ref_y->template get_vector<T>();

            // inc_y will hold f(x+dx) values
            auto inc_y = backend->make_primary_tensor_view<T>(y_shape);

            // Assuming vars, y, and results are row-major

            T inv_delta = 1 / delta;

            size_t pos = 0;

            for (size_t i = 0; i < args.size(); ++i)
            {
                if (std::find(indep_params.begin(), indep_params.end(), params[i]) !=
                    indep_params.end())
                {
                    auto arg = args[i];
                    auto res = results[pos]->get_vector<T>();
                    auto vec = arg->get_vector<T>();
                    for (size_t j = 0; j < vec.size(); j++)
                    {
                        auto old_val = vec[j];
                        vec[j] += delta;
                        arg->write(vec);
                        cf->tensor_call(args, {inc_y});
                        auto inc_vec = inc_y->template get_vector<T>();
                        vec[j] = old_val;
                        arg->write(vec);
                        size_t res_k = j;
                        for (size_t k = 0; k < inc_vec.size(); k++)
                        {
                            auto y1 = inc_vec[k];
                            auto y0 = ref_vec[k];
                            res[res_k] = inv_delta * (y1 - y0);
                            res_k += vec.size();
                        }
                    }
                    results[pos]->write(res);
                    pos++;
                }
            }
            return results;
        }
    }
}
