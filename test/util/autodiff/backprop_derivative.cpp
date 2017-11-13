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

#include <memory>
#include <vector>

#include "backprop_derivative.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/types/type.hpp"

using namespace ngraph;

template <typename ET>
std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>
    autodiff::backprop_derivative(const std::shared_ptr<runtime::Manager>& manager,
                                  const std::shared_ptr<runtime::Backend>& backend,
                                  const std::shared_ptr<Function>& f,
                                  const std::vector<std::shared_ptr<runtime::TensorView>>& args,
                                  const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
{
    auto y = f->get_result();
    Shape y_shape =
        std::dynamic_pointer_cast<const TensorViewType>(y->get_value_type())->get_shape();

    auto c_param = std::make_shared<op::Parameter>(ET::element_type(), y_shape);
    auto c_arg = backend->make_parameterized_tensor_view<ET>(y_shape);
    auto params = f->get_parameters();

    std::vector<std::shared_ptr<Node>> deriv_nodes;
    std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> bprops;
    std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> results;

    for (auto param : indep_params)
    {
        Shape s = y_shape;
        auto param_shape =
            std::dynamic_pointer_cast<const TensorViewType>(param->get_value_type())->get_shape();
        s.insert(s.end(), param_shape.begin(), param_shape.end());
        results.push_back(backend->make_parameterized_tensor_view<ET>(s));
        bprops.push_back(backend->make_parameterized_tensor_view<ET>(param_shape));
        deriv_nodes.push_back(y->backprop_node(param, c_param));
    }

    std::vector<std::shared_ptr<op::Parameter>> df_params = params;
    df_params.push_back(c_param);
    auto df_result = std::make_shared<op::Tuple>(deriv_nodes);
    auto df = std::make_shared<Function>(df_result, df_result->get_value_type(), df_params);

    auto external = manager->compile(df);
    auto cf = backend->make_call_frame(external);

    // We compute the derivatives chunk by chunk
    std::vector<typename std::vector<typename ET::type>::iterator> result_pos;
    for (auto result : results)
    {
        result_pos.push_back(result->get_vector().begin());
    }

    ngraph::runtime::TensorViewPtrs args_tv;
    args_tv.insert(args_tv.begin(), args.begin(), args.end());
    args_tv.push_back(c_arg);

    runtime::TensorViewPtrs bprops_tv;
    bprops_tv.insert(bprops_tv.begin(), bprops.begin(), bprops.end());

    auto& c_vec = c_arg->get_vector();
    for (size_t i = 0; i < c_vec.size(); i++)
    {
        c_vec[i] = 1;
        cf->tensor_call(args_tv, bprops_tv);
        c_vec[i] = 0;
        for (size_t j = 0; j < results.size(); j++)
        {
            auto& bprop_vec = bprops[j]->get_vector();
            result_pos[j] = std::copy(bprop_vec.begin(), bprop_vec.end(), result_pos[j]);
        }
    }

    return results;
}

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>>
    autodiff::backprop_derivative<ngraph::element::Float32>(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<Function>& f,
        const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args,
        const std::vector<std::shared_ptr<op::Parameter>>& indep_params);

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>>
    autodiff::backprop_derivative<ngraph::element::Float64>(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<Function>& f,
        const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args,
        const std::vector<std::shared_ptr<op::Parameter>>& indep_params);
