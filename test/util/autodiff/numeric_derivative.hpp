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
        /// @returns vector of dy/dvar, where each dy/dvar's shape is concat(y.shape(), var.shape())
        template <typename ET>
        std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> numeric_derivative(
            const std::shared_ptr<runtime::Manager>& manager,
            const std::shared_ptr<runtime::Backend>& backend,
            const std::shared_ptr<Function>& f,
            const std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>>& args,
            typename ET::type delta)
        {
            auto y = f->get_result();

            Shape y_shape =
                std::dynamic_pointer_cast<const TensorViewType>(y->get_value_type())->get_shape();

            auto params = f->get_parameters();

            // Results for each derivative, shape Y|X_i
            std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> results;
            for (auto param : params)
            {
                Shape s = y_shape;
                auto param_shape =
                    std::dynamic_pointer_cast<const TensorViewType>(param->get_value_type())
                        ->get_shape();
                s.insert(s.end(), param_shape.begin(), param_shape.end());
                results.push_back(backend->make_parameterized_tensor_view<ET>(s));
            }

            auto external = manager->compile(f);
            auto cf = backend->make_call_frame(external);

            // ref_y is the function evaluated at the args
            auto ref_y = backend->make_parameterized_tensor_view<ET>(y_shape);

            ngraph::runtime::TensorViewPtrs args_tv;
            args_tv.insert(args_tv.begin(), args.begin(), args.end());

            cf->tensor_call(args_tv, runtime::TensorViewPtrs{ref_y});
            auto& ref_vec = ref_y->get_vector();

            // inc_y will hold f(x+dx) values
            auto inc_y = backend->make_parameterized_tensor_view<ET>(y_shape);
            auto& inc_vec = inc_y->get_vector();

            // Assuming vars, y, and results are row-major

            typename ET::type inv_delta = 1 / delta;
            for (size_t i = 0; i < args.size(); ++i)
            {
                auto arg = args[i];
                auto res = results[i]->get_vector();
                auto vec = arg->get_vector();
                for (size_t j = 0; j < vec.size(); j++)
                {
                    auto old_val = vec[j];
                    vec[j] += delta;
                    arg->write(vec);
                    cf->tensor_call(args_tv, {inc_y});
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
                results[i]->write(res);
            }
            return results;
        }
    }
}
