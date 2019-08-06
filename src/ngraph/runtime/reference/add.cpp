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

#include "ngraph/runtime/reference/add.hpp"

using namespace std;
using namespace ngraph;

void runtime::reference::add(const TensorBase& arg0, const TensorBase& arg1, TensorBase& out)
{
    NGRAPH_CHECK(arg0.element_type() == arg1.element_type() &&
                 arg0.element_type() == out.element_type());
    NGRAPH_CHECK(arg0.shape() == arg1.shape() && arg0.shape() == out.shape());
    WITH_ET(arg0.element_type(), T, {
        add<T>(arg0.buffer<T>(), arg1.buffer<T>(), out.buffer<T>(), shape_size(arg0.shape()));
    });
}
