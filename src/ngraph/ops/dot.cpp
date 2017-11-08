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

#include <functional>
#include <memory>

#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::Dot::Dot(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
    : TensorViewArgs({arg0, arg1})
{
    auto arg0_tensor_type = get_inputs().at(0).get_tensor_view_type();
    auto arg1_tensor_type = get_inputs().at(1).get_tensor_view_type();

    if (arg0_tensor_type->get_element_type() != arg1_tensor_type->get_element_type())
    {
        throw ngraph_error("Arguments to dot must have the same element type");
    }

    vector<size_t> arg0_shape = arg0_tensor_type->get_shape();
    vector<size_t> arg1_shape = arg1_tensor_type->get_shape();
    size_t arg0_reduction = arg0_shape.size() - 1;
    size_t arg1_reduction;
    const bool is_scalar_mult = arg0_shape.size() == 0 || arg1_shape.size() == 0;

    if (arg1_shape.size() > 1)
    {
        arg1_reduction = arg1_shape.size() - 2;
    }
    else
    {
        arg1_reduction = arg1_shape.size() - 1;
    }
    if (!is_scalar_mult && (arg0_shape.at(arg0_reduction) != arg1_shape.at(arg1_reduction)))
    {
        throw ngraph_error("Dot reduction axes not compatible");
    }

    vector<size_t> result_shape;
    result_shape.reserve(arg0_shape.size() + arg1_shape.size() - (is_scalar_mult ? 0 : 2));

    for (auto i = 0; i < arg0_shape.size(); i++)
    {
        if (is_scalar_mult || i != arg0_reduction)
        {
            result_shape.push_back(arg0_shape[i]);
        }
    }

    for (auto i = 0; i < arg1_shape.size(); i++)
    {
        if (is_scalar_mult || i != arg1_reduction)
        {
            result_shape.push_back(arg1_shape[i]);
        }
    }

    auto result_type =
        make_shared<TensorViewType>(arg0_tensor_type->get_element_type(), result_shape);
    set_value_type_checked(result_type);
}

template <typename T>
T range(size_t n);

template <>
ngraph::AxisSet range<ngraph::AxisSet>(size_t n)
{
    ngraph::AxisSet result;
    for (size_t i = 0; i < n; i++)
    {
        result.insert(i);
    }
    return result;
}

template <>
ngraph::AxisVector range<ngraph::AxisVector>(size_t n)
{
    ngraph::AxisVector result;
    for (size_t i = 0; i < n; i++)
    {
        result.push_back(i);
    }
    return result;
}

void op::Dot::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto x = m_arguments[0];
    auto y = m_arguments[1];

    auto x_shape = x->get_shape();
    auto y_shape = y->get_shape();
    auto delta_shape = delta->get_shape();

    if (is_scalar(x_shape))
    {
        adjoints.add_delta(y, make_shared<Dot>(delta, x));
        if (is_scalar(y_shape))
        {
            // Just multiplication
            adjoints.add_delta(x, delta * y);
            return;
        }
        // scale dot tensor
        adjoints.add_delta(x, make_shared<Sum>(delta * y, range<AxisSet>(y_shape.size())));
        return;
    }
    if (is_scalar(y_shape))
    {
        // tensor dot scalar
        adjoints.add_delta(x, make_shared<Dot>(delta, y));
        adjoints.add_delta(y, make_shared<Sum>(delta * x, range<AxisSet>(x_shape.size())));
        return;
    }
    if (is_vector(y_shape))
    {
        if (is_vector(x_shape))
        {
            adjoints.add_delta(x, make_shared<Dot>(delta, y));
        }
        else
        {
            // X has shape IJ, Y has shape J, delta has shape I
            // delta -> (I, 1)
            // Y -> (1, J)
            // delta . Y is (I, J)
            Shape shape_delta_1 = delta->get_shape();
            shape_delta_1.push_back(1);
            auto delta_1 =
                make_shared<Broadcast>(delta, shape_delta_1, AxisSet{delta->get_shape().size()});
            Shape shape_1_y{1};
            shape_1_y.insert(shape_1_y.end(), y_shape.begin(), y_shape.end());
            auto y_1 = make_shared<Broadcast>(y, shape_1_y, AxisSet{0});
            adjoints.add_delta(x, make_shared<Dot>(delta_1, y_1));
        }
        // X has shape IJ
        // Y has shape J
        // delta has shape I
        // Need to move J to front of X and multiply by Y
        Shape shape_xt(x_shape.size());
        AxisVector x_axes(x_shape.size());
        shape_xt[0] = x_shape.at(x_shape.size() - 1);
        x_axes[0] = x_shape.size() - 1;
        for (size_t i = 1; i < x_shape.size(); ++i)
        {
            shape_xt[i] = x_shape[i - 1];
            x_axes[i] = i - 1;
        }
        auto x_reshape = make_shared<Reshape>(x, x_axes, shape_xt);
        adjoints.add_delta(y, make_shared<Dot>(x_reshape, delta));
        return;
    }
    // Tensor tensor case
    // X is Ij
    // Y = Kjl
    // X.Y, delta is IKl
    //
    // delta -> I(Kl)
    // Y -> (Kl)j
    // delta.Y -> Ij
    Shape s_I;
    s_I.insert(s_I.begin(), x_shape.begin(), x_shape.end() - 1);
    size_t s_j = x_shape[x_shape.size() - 1];
    Shape s_K;
    s_K.insert(s_K.begin(), y_shape.begin(), y_shape.end() - 2);
    size_t s_l = y_shape[y_shape.size() - 1];
    size_t s_Kl = shape_size(s_K) * s_l;

    Shape shape_delta_I_Kl;
    shape_delta_I_Kl.insert(shape_delta_I_Kl.end(), s_I.begin(), s_I.end());
    shape_delta_I_Kl.push_back(s_Kl);
    AxisVector idx_delta_I_Kl = range<AxisVector>(delta_shape.size());
    auto delta_I_Kl = make_shared<Reshape>(delta, idx_delta_I_Kl, shape_delta_I_Kl);

    Shape shape_y_Kl_j{s_Kl, s_j};
    AxisVector idx_y_Kl_j = range<AxisVector>(y_shape.size() - 2);
    idx_y_Kl_j.push_back(y_shape.size() - 1);
    idx_y_Kl_j.push_back(y_shape.size() - 2);
    auto y_Kl_j = make_shared<Reshape>(y, idx_y_Kl_j, shape_y_Kl_j);
    adjoints.add_delta(x, make_shared<Dot>(delta_I_Kl, y_Kl_j));

    // delta -> K(I)l
    // X -> j(I)
    // X.delta -> jKl -> Kjl
    Shape shape_delta_K_I_l;
    shape_delta_K_I_l.insert(shape_delta_K_I_l.begin(), s_K.begin(), s_K.end());
    shape_delta_K_I_l.push_back(shape_size(s_I));
    shape_delta_K_I_l.push_back(s_l);
    AxisVector idx_delta = range<AxisVector>(delta_shape.size());
    AxisVector idx_delta_K_I_l;
    idx_delta_K_I_l.insert(idx_delta_K_I_l.end(),
                           idx_delta.begin() + s_I.size(),
                           idx_delta.begin() + s_I.size() + s_K.size());
    idx_delta_K_I_l.insert(
        idx_delta_K_I_l.end(), idx_delta.begin(), idx_delta.begin() + s_I.size());
    idx_delta_K_I_l.push_back(delta_shape.size() - 1);
    auto delta_K_I_l = make_shared<Reshape>(delta, idx_delta_K_I_l, shape_delta_K_I_l);

    Shape shape_x_j_I;
    shape_x_j_I.push_back(s_j);
    shape_x_j_I.push_back(shape_size(s_I));
    AxisVector idx_x = range<AxisVector>(x_shape.size());
    AxisVector idx_x_j_I;
    idx_x_j_I.push_back(idx_x[idx_x.size() - 1]);
    idx_x_j_I.insert(idx_x_j_I.end(), idx_x.begin(), idx_x.begin() + idx_x.size() - 1);
    auto x_j_I = make_shared<Reshape>(x, idx_x_j_I, shape_x_j_I);
    auto jKl = make_shared<Dot>(x_j_I, delta_K_I_l);
    Shape shape_Kjl;
    shape_Kjl.insert(shape_Kjl.end(), s_K.begin(), s_K.end());
    shape_Kjl.push_back(s_j);
    shape_Kjl.push_back(s_l);
    AxisVector idx_Kjl;
    for (size_t i = 1; i < s_K.size() + 1; ++i)
    {
        idx_Kjl.push_back(i);
    }
    idx_Kjl.push_back(0);
    idx_Kjl.push_back(y_shape.size() - 1);
    auto Klj = make_shared<Reshape>(jKl, idx_Kjl, shape_Kjl);
    adjoints.add_delta(y, Klj);
}
