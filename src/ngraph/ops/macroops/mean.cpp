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

#include "ngraph/ops/macroops/mean.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include "ngraph/function.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/util.hpp"

using namespace ngraph::op;

std::shared_ptr<::ngraph::Node> Mean::lower()
{
    auto arg = m_arguments.at(0);

    auto st = get_shape_et(m_arguments.at(0));
    size_t num_dims = st.shape.size();

    if (std::any_of(begin(m_reduction_axes), end(m_reduction_axes), [num_dims](size_t axis) {
            return axis >= num_dims;
        }))
    {
        throw ngraph_error("Reduction axis for mean is out of bounds");
    }

    auto sum = std::make_shared<Sum>(arg, m_reduction_axes);
    assert(sum->get_value_type());
    auto sum_tensor_view_type =
        std::dynamic_pointer_cast<const TensorViewType>(sum->get_value_type());
    assert(sum_tensor_view_type);

    size_t n = std::accumulate(begin(m_reduction_axes),
                               end(m_reduction_axes),
                               static_cast<size_t>(1u),
                               [](size_t a, size_t b) { return a * b; });

    auto constant = std::make_shared<Constant>(sum_tensor_view_type->get_element_type(),
                                               sum_tensor_view_type->get_shape(),
                                               std::to_string(n));
    return sum / constant;
}
