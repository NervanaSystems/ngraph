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

#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/sign.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::op;

void Sign::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto x = m_arguments[0];

    auto x_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(x->get_value_type());
    auto& x_element_type = x_tensor_view_type->get_element_type();
    auto x_shape = x_tensor_view_type->get_shape();
    auto x_zero = std::make_shared<op::Constant>(x_element_type, x_shape, "0");

    adjoints.add_delta(x, x_zero);
}
