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

#include "ngraph/runtime/reference/result.hpp"

using namespace std;
using namespace ngraph;

void runtime::reference::result(const TensorBase& arg, TensorBase& out)
{
    NGRAPH_CHECK(arg.element_type() == out.element_type());
    NGRAPH_CHECK(arg.shape() == out.shape());
    WITH_ET(arg.element_type(), T, {
        result<T>(arg.buffer<T>(), out.buffer<T>(), shape_size(arg.shape()));
    });
}
