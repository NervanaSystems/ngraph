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
#include "util/test_tools.hpp"

namespace ngraph
{
    class Node;
    class Function;

    namespace runtime
    {
        class Backend;
        class Manager;
    } // namespace runtime

    namespace autodiff
    {
        template <typename T>
        std::vector<std::shared_ptr<runtime::TensorView>>
            backprop_derivative(const std::shared_ptr<runtime::Manager>& manager,
                                const std::shared_ptr<runtime::Backend>& backend,
                                const std::shared_ptr<Function>& f,
                                const std::vector<std::shared_ptr<runtime::TensorView>>& args,
                                const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            // y = f(X)
            // using X (upper case) to denote all paramenters of f
            // using x (lower case) to denote an individual paramemter of f a.k.a. Xj
            // NOTE: using X* to denote all x "of interest" represented by indep_params
            Shape y_shape = f->get_output_shape(0);

            // adjoint
            auto c_param = std::make_shared<op::Parameter>(element::from<T>(), y_shape);
            auto c_arg = backend->make_primary_tensor_view<T>(y_shape);

            // df/dX*
            // return value for f'(X, c)
            std::vector<std::shared_ptr<Node>> df_output_params;
            std::vector<std::shared_ptr<runtime::TensorView>> df_output_args;

            // return value for this function
            std::vector<std::shared_ptr<runtime::TensorView>> results;

            // for each x "of interest"
            for (auto x : indep_params)
            {
                auto x_shape = x->get_shape();

                // each element of y has a derivative with respect to each element of x
                // hence, create a y by x sized tensor for this result
                auto y_by_x_shape = y_shape;
                y_by_x_shape.insert(y_by_x_shape.end(), x_shape.begin(), x_shape.end());
                results.push_back(backend->make_primary_tensor_view<T>(y_by_x_shape));

                // add df/dx to df/dX*
                df_output_params.push_back(f->get_output_op(0)->backprop_node(x, c_param));
                df_output_args.push_back(backend->make_primary_tensor_view<T>(x_shape));
            }

            // (X, c)
            // input to f'(X, c)
            std::vector<std::shared_ptr<op::Parameter>> df_input_params = f->get_parameters();
            df_input_params.push_back(c_param);

            // df/dX* = f'(X, c)
            auto df = std::make_shared<Function>(df_output_params, df_input_params);

            // create fprop cache
            // creates modified forward function -> (y, cached) = f(x)
            // creates modified backward function -> df/dX* = f'(c, cached)
            auto fprop_cache = cache_fprop(f, df, {c_param});

            // modified f outputs
            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> f_output_args;
            f_output_args.push_back(backend->make_primary_tensor_view<T>(y_shape));

            // modified f' inputs
            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> df_input_args;
            df_input_args.push_back(c_arg);

            // add cached nodes to both modified f outputs and modified f' inputs
            for (auto node : fprop_cache.fprop_output_nodes)
            {
                auto tv = backend->make_primary_tensor_view<T>(node->get_shape());
                df_input_args.push_back(tv);
                f_output_args.push_back(tv);
            }

            // compile and run modified (y, cached) = f(x)
            auto cache_fwd = manager->compile(fprop_cache.fprop);
            auto cache_fwd_cf = backend->make_call_frame(cache_fwd);
            cache_fwd_cf->tensor_call(args, f_output_args);

            // compile modified df/dX* = f'(c, cached)
            auto external = manager->compile(fprop_cache.bprop);
            auto cf = backend->make_call_frame(external);

            // create storage for results
            // * outer vector size = number of x "of interest"
            // * inner vector size = number of elements in y * number of elements in x
            std::vector<std::vector<T>> result_vect;
            std::vector<typename std::vector<T>::iterator> result_pos;
            for (auto result : results)
            {
                result_vect.push_back(read_vector<T>(result)); // storage for results
                result_pos.push_back(result_vect.back().begin());
            }

            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> args_tv;
            args_tv.insert(args_tv.begin(), args.begin(), args.end());
            args_tv.push_back(c_arg);

            std::vector<std::shared_ptr<ngraph::runtime::TensorView>> bprops_tv;
            bprops_tv.insert(bprops_tv.begin(), bprops.begin(), bprops.end());
			
            // get adjoint and force to all elements to zero
            auto c_vec = read_vector<T>(c_arg);
            fill(c_vec.begin(), c_vec.end(), 0);

            // for each element of the adjoint
            // same as saying for each element of y
            for (size_t i = 0; i < c_vec.size(); i++)
            {
                // set a single adjoint element
                c_vec[i] = 1;
                write_vector(c_arg, c_vec);

                // call modified df/dX* = f'(c, cached)
                cf->tensor_call(df_input_args, df_output_args);

                // reset the adjoint element
                c_vec[i] = 0;
                write_vector(c_arg, c_vec);

                // for each result
                // same as saying for each x "of interest"
                for (size_t j = 0; j < results.size(); j++)
                {
                    // copy df/dx to storage for this element of y
                    auto bprop_vec = read_vector<T>(bprops[j]);
                    result_pos[j] = std::copy(bprop_vec.begin(), bprop_vec.end(), result_pos[j]);
                }
            }

            // copy storage to results and return
            for (size_t j = 0; j < results.size(); j++)
            {
                write_vector(results[j], result_vect[j]);
            }
            return results;
        }
    } // namespace autodiff
} // namespace ngraph
