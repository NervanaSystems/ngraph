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

#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

void runtime::reference::convert(const TensorValue& arg, TensorValue& out)
{
    WITH_ET(arg.element_type(), Targ, {
        WITH_ET(out.element_type(), Tout, {
            convert(arg.buffer<Targ>(), out.buffer<Tout>(), shape_size(arg.shape()));
        });
    });
}
