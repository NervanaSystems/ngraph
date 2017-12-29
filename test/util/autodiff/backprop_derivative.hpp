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

#include "ngraph/log.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    class Node;
    class Function;

    namespace runtime
    {
        class Backend;
        class Manager;
    }

    namespace autodiff
    {
        /// @brief Returns a FunctionSpec for the backprop derivative of its argument.
        /// @param f is f(X_i...)
        /// @returns f'(X_i..., c) where f'(x_i, ..., c)_j is backprop for X_j
        std::shared_ptr<Function> backprop_function(const std::shared_ptr<Function>& f);

        template <typename T>
        std::vector<std::shared_ptr<runtime::TensorView>>
            backprop_derivative(const std::shared_ptr<runtime::Manager>& manager,
                                const std::shared_ptr<runtime::Backend>& backend,
                                const std::shared_ptr<Function>& f,
                                const std::vector<std::shared_ptr<runtime::TensorView>>& args,
                                const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            Shape y_shape = f->get_output_shape(0);

            auto c_param = std::make_shared<op::Parameter>(element::from<T>(), y_shape);
            auto c_arg = backend->make_primary_tensor_view<T>(y_shape);
            auto params = f->get_parameters();

            std::vector<std::shared_ptr<Node>> deriv_nodes;
            std::vector<std::shared_ptr<runtime::TensorView>> bprops;
            std::vector<std::shared_ptr<runtime::TensorView>> results;

            for (auto param : indep_params)
            {
                Shape s = y_shape;
                auto param_shape = param->get_shape();
                s.insert(s.end(), param_shape.begin(), param_shape.end());
                results.push_back(backend->make_primary_tensor_view<T>(s));
                bprops.push_back(backend->make_primary_tensor_view<T>(param_shape));
                deriv_nodes.push_back(f->get_output_op(0)->backprop_node(param, c_param));
            }

            std::vector<std::shared_ptr<op::Parameter>> df_params = params;
            df_params.push_back(c_param);
            auto df = std::make_shared<Function>(deriv_nodes, df_params);

            auto external = manager->compile(df);
            auto cf = backend->make_call_frame(external);

            // We compute the derivatives chunk by chunk
            std::vector<typename std::vector<T>::iterator> result_pos;
            std::vector<std::vector<T>> result_vect;
            for (auto result : results)
            {
                result_vect.push_back(result->get_vector<T>()); // storage for results
                result_pos.push_back(result_vect.back().begin());
            }

            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> args_tv;
            args_tv.insert(args_tv.begin(), args.begin(), args.end());
            args_tv.push_back(c_arg);

            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> bprops_tv;
            bprops_tv.insert(bprops_tv.begin(), bprops.begin(), bprops.end());

            auto c_vec = c_arg->template get_vector<T>();
            fill(c_vec.begin(), c_vec.end(), 0);
            for (size_t i = 0; i < c_vec.size(); i++)
            {
                c_vec[i] = 1;
                c_arg->write(c_vec);
                cf->tensor_call(args_tv, bprops_tv);
                c_vec[i] = 0;
                c_arg->write(c_vec);
                for (size_t j = 0; j < results.size(); j++)
                {
                    auto bprop_vec = bprops[j]->get_vector<T>();
                    result_pos[j] = std::copy(bprop_vec.begin(), bprop_vec.end(), result_pos[j]);
                }
            }

            // Copy results from temp to result vector
            for (size_t j = 0; j < results.size(); j++)
            {
                results[j]->write(result_vect[j]);
            }
            return results;
        }
    }
}
