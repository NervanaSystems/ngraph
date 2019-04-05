//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/builder/quantization/quantized_linear_dot.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/experimental/quantized_dot.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            shared_ptr<Node> QuantizedDotInteger(shared_ptr<Node> input, shared_ptr<Node> filter)
            {
                auto output_scale = make_constant(element::f32, Shape{}, 1);
                return make_shared<op::QuantizedDot>(input, filter, output_scale, false, false);
            }
        }
    }
}
