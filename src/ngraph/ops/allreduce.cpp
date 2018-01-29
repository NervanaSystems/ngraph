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

#ifdef NGRAPH_DISTRIBUTED

#include "ngraph/ops/allreduce.hpp"

using namespace std;
using namespace ngraph;

op::AllReduce::AllReduce(const std::shared_ptr<Node>& arg)
    : RequiresTensorViewArgs("AllReduce", {arg})
{
    auto& input = m_inputs.at(0);
    set_value_type_checked(
        make_shared<TensorViewType>(input.get_element_type(), input.get_shape()));
}

#endif
