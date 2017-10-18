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

#include <algorithm>
#include <cassert>
#include <cmath>

#include "ngraph/function.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/utils.hpp"

std::shared_ptr<ngraph::runtime::Tuple> ngraph::runtime::make_tuple(
    const std::vector<std::shared_ptr<ngraph::runtime::Value>>& elements)
{
    return std::make_shared<ngraph::runtime::Tuple>(elements);
}

template <typename ET>
bool ngraph::runtime::all_close(
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>& a,
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>& b,
    typename ET::type rtol,
    typename ET::type atol)
{
    // Check that the layouts are compatible
    if (*a->get_tensor_view_layout() != *b->get_tensor_view_layout())
    {
        throw ngraph_error("Cannot compare tensors with different layouts");
    }

    if (a->get_shape() != b->get_shape())
        return false;

    return ngraph::runtime::all_close(a->get_vector(), b->get_vector(), rtol, atol);
}

template bool ngraph::runtime::all_close<ngraph::element::Float32>(
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& a,
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& b,
    ngraph::element::Float32::type rtol,
    ngraph::element::Float32::type atol);

template bool ngraph::runtime::all_close<ngraph::element::Float64>(
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>& a,
    const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>& b,
    ngraph::element::Float64::type rtol,
    ngraph::element::Float64::type atol);

template <typename T>
bool ngraph::runtime::all_close(const std::vector<T>& a, const std::vector<T>& b, T rtol, T atol)
{
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
        {
            return false;
        }
    }
    return true;
}

template bool ngraph::runtime::all_close<float>(const std::vector<float>& a,
                                                const std::vector<float>& b,
                                                float rtol,
                                                float atol);

template bool ngraph::runtime::all_close<double>(const std::vector<double>& a,
                                                 const std::vector<double>& b,
                                                 double rtol,
                                                 double atol);

ngraph::runtime::FunctionSpec::operator std::shared_ptr<Function>() const
{
    return std::make_shared<ngraph::Function>(m_result, m_result_type, m_parameters);
}

// Returns (dy/(dXs))(C, Xs)
std::shared_ptr<ngraph::runtime::FunctionSpec>
    ngraph::runtime::derivative(const std::shared_ptr<ngraph::runtime::FunctionSpec>& f)
{
    auto Y = f->get_result();
    auto Xs = f->get_parameters();
    auto Y_tv_type = std::dynamic_pointer_cast<const ngraph::TensorViewType>(Y->get_value_type());
    auto C = std::make_shared<ngraph::op::Parameter>(Y_tv_type->get_element_type(),
                                                     Y_tv_type->get_shape());
    std::vector<std::shared_ptr<ngraph::Node>> dYdXs(Xs.size());
    transform(Xs.begin(), Xs.end(), dYdXs.begin(), [C, Y](const std::shared_ptr<ngraph::Node>& X) {
        return Y->backwards_derivative(X, C);
    });
    auto result = std::make_shared<ngraph::op::Tuple>(dYdXs);
    std::vector<std::shared_ptr<ngraph::op::Parameter>> args;
    args.push_back(C);
    args.insert(args.end(), Xs.begin(), Xs.end());
    return std::make_shared<ngraph::runtime::FunctionSpec>(result, result->get_value_type(), args);
}

template <typename ET>
std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>
    ngraph::runtime::numeric_derivative(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>& args,
        typename ET::type delta)
{
    auto y = f->get_result();

    Shape y_shape =
        std::dynamic_pointer_cast<const ngraph::TensorViewType>(y->get_value_type())->get_shape();

    // Check all the shapes
    std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>> results;
    for (size_t i = 0; i < args.size(); i++)
    {
        Shape s = y_shape;
        auto arg_shape = args[i]->get_shape();
        s.insert(s.end(), arg_shape.begin(), arg_shape.end());
        results.push_back(backend->make_parameterized_tensor_view<ET>(s));
    }

    auto external = manager->compile(*f);
    auto cf = backend->make_call_frame(external);

    // ref_y is the function evaluated at the args
    auto ref_y = backend->make_parameterized_tensor_view<ET>(y_shape);

    ngraph::runtime::TensorViewPtrs args_tv;
    args_tv.insert(args_tv.begin(), args.begin(), args.end());

    cf->tensor_call(args_tv, TensorViewPtrs{ref_y});
    auto ref_vec = ref_y->get_vector();

    // inc_y will hold f(x+dx) values
    auto inc_y = backend->make_parameterized_tensor_view<ET>(y_shape);
    auto inc_vec = inc_y->get_vector();

    // Assuming vars, y, and results are row-major

    typename ET::type inv_delta = 1 / delta;
    for (size_t i = 0; i < args.size(); ++i)
    {
        auto arg = args[i];
        auto df_darg = results[i];
        auto df_darg_it = df_darg->get_vector().begin();
        std::vector<typename ET::type>& vec = arg->get_vector();
        for (size_t j = 0; j < vec.size(); i++)
        {
            auto old_val = vec[j];
            vec[j] += delta;
            cf->tensor_call(args_tv, {inc_y});
            vec[j] = old_val;
            df_darg_it = std::transform(inc_vec.begin(),
                                        inc_vec.end(),
                                        ref_vec.begin(),
                                        df_darg_it,
                                        [inv_delta](typename ET::type y1, typename ET::type y0) {
                                            return inv_delta * (y1 - y0);
                                        });
        }
    }
    return results;
}

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>>
    ngraph::runtime::numeric_derivative(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float32>>>& args,
        element::Float32::type delta);

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>>
    ngraph::runtime::numeric_derivative(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float64>>>& args,
        element::Float64::type delta);

template <typename ET>
std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>
    ngraph::runtime::backwards_derivative(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>& args)
{
    auto y = f->get_result();
    Shape y_shape =
        std::dynamic_pointer_cast<const ngraph::TensorViewType>(y->get_value_type())->get_shape();

    auto c_param = std::make_shared<op::Parameter>(ET::element_type(), y_shape);
    auto c_arg = backend->make_parameterized_tensor_view<ET>(y_shape);
    auto params = f->get_parameters();

    std::vector<std::shared_ptr<ngraph::Node>> deriv_nodes;
    std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>> bprops;
    std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>> results;
    for (auto param : params)
    {
        Shape s = y_shape;
        auto param_shape =
            std::dynamic_pointer_cast<const ngraph::TensorViewType>(param->get_value_type())
                ->get_shape();
        s.insert(s.end(), param_shape.begin(), param_shape.end());
        results.push_back(backend->make_parameterized_tensor_view<ET>(s));
        bprops.push_back(backend->make_parameterized_tensor_view<ET>(param_shape));
        deriv_nodes.push_back(y->backwards_derivative(param, c_param));
    }

    std::vector<std::shared_ptr<op::Parameter>> df_params = params;
    df_params.push_back(c_param);
    auto df_result = std::make_shared<op::Tuple>(deriv_nodes);
    auto df = std::make_shared<ngraph::Function>(df_result, df_result->get_value_type(), df_params);

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

    TensorViewPtrs bprops_tv;
    bprops_tv.insert(bprops_tv.begin(), bprops.begin(), bprops.end());

    auto c_vec = c_arg->get_vector();
    for (size_t i = 0; i < c_vec.size(); i++)
    {
        c_vec[i] = 1;
        cf->tensor_call(args_tv, bprops_tv);
        c_vec[i] = 0;
        for (size_t j = 0; j < results.size(); j++)
        {
            auto bprop_vec = bprops[j]->get_vector();
            result_pos[j] =
                results[j]->get_vector().insert(result_pos[j], bprop_vec.begin(), bprop_vec.end());
        }
    }

    return results;
}

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>>
    ngraph::runtime::backwards_derivative<ngraph::element::Float32>(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float32>>>& args);

template std::vector<
    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>>
    ngraph::runtime::backwards_derivative<ngraph::element::Float64>(
        const std::shared_ptr<runtime::Manager>& manager,
        const std::shared_ptr<runtime::Backend>& backend,
        const std::shared_ptr<FunctionSpec>& f,
        const std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float64>>>& args);
