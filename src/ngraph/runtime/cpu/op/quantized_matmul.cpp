//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <functional>
#include <memory>
#include <utility>

#include "ngraph/shape.hpp"
#include "quantized_matmul.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedMatmul::QuantizedMatmul(const shared_ptr<Node>& data,
                                     const shared_ptr<Node>& weights,
                                     const shared_ptr<Node>& scale,
                                     bool requantize,
                                     bool with_relu)
    : Op("QuantizedMatmul", check_single_output_args({data, weights, scale}))
    , m_requantize(requantize)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    auto& data_shape = data->get_shape();
    auto& weights_shape = weights->get_shape();
    // QuantizedMatmul does [n, ic] * [oc, ic] = [n, oc]
    NODE_VALIDATION_CHECK(this,
                          data_shape.size() == 2 && weights_shape.size() == 2 &&
                              data_shape[1] == weights_shape[1],
                          "only valid tensors of rank 2 supported. data shape ",
                          data_shape,
                          " weights shape ",
                          weights_shape);

    auto output_et = requantize ? (with_relu ? element::u8 : element::i8) : element::i32;
    set_output_type(0, output_et, Shape{data_shape[0], weights_shape[0]});
}
