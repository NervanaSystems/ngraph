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

const string op::QuantizedMatmul::type_name{"QuantizedMatmul"};

op::QuantizedMatmul::QuantizedMatmul(const Output<Node>& data,
                                     const Output<Node>& weights,
                                     const Output<Node>& scale,
                                     const element::Type& output_type)
    : Op({data, weights, scale})
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();

    auto& data_shape = data.get_shape();
    auto& weights_shape = weights.get_shape();
    // QuantizedMatmul does [n, ic] * [oc, ic] = [n, oc]
    NODE_VALIDATION_CHECK(this,
                          data_shape.size() == 2 && weights_shape.size() == 2 &&
                              data_shape[1] == weights_shape[1],
                          "only valid tensors of rank 2 supported. data shape ",
                          data_shape,
                          " weights shape ",
                          weights_shape);

    set_output_type(0, output_type, Shape{data_shape[0], weights_shape[0]});
}
