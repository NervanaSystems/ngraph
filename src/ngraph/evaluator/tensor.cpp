//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <utility>

#include "ngraph/evaluator/tensor.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/minimum.hpp"

using namespace std;
using namespace ngraph;

evaluator::Tensor::Tensor(const std::shared_ptr<ngraph::op::Constant>& c)
    : Tensor(c->output(0).get_element_type(), c->output(0).get_shape(), c->get_data_ptr())
{
}

vector<evaluator::Tensor> exec_constant(shared_ptr<Node> node, vector<evaluator::Tensor>& inputs)
{
    auto op = as_type_ptr<op::Constant>(node);
    auto shape = op->get_output_partial_shape(0);
    if (shape.rank().is_static() && shape.rank().get_length() == 0)
    {
        auto et = op->get_output_element_type(0);
        if (et == element::i64)
        {
            vector<int64_t> elts = op->get_vector<int64_t>();
            return {evaluator::Tensor(op)};
        }
    }
    return {evaluator::Tensor()};
}

vector<evaluator::Tensor> exec_minimum(shared_ptr<Node> node, vector<evaluator::Tensor>& inputs)
{
    auto op = as_type_ptr<op::Minimum>(node);
    auto shape = op->get_output_partial_shape(0);
    if (shape.rank().is_static() && shape.rank().get_length() == 0)
    {
        auto et = op->get_output_element_type(0);
        if (et == element::i64)
        {
            int64_t min_value = numeric_limits<int64_t>::max();
            {
                auto& v1 = inputs.at(0);
                if (v1.get_element_type() == et)
                {
                    min_value = std::min(min_value, v1.get_read_data<element::Type_t::i64>()[0]);
                }
            }
            {
                auto& v2 = inputs.at(1);
                if (v2.get_element_type() == et)
                {
                    min_value = std::min(min_value, v2.get_read_data<element::Type_t::i64>()[0]);
                }
            }
            return {min_value == numeric_limits<int64_t>::max()
                        ? evaluator::Tensor()
                        : evaluator::Tensor(et, shape.to_shape())
                              .set_elements<element::Type_t::i64>({min_value})};
        }
    }
    return {evaluator::Tensor()};
}
