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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace runtime;

CallFrame::CallFrame(Function&                                              function,
                     const std::vector<std::shared_ptr<PrimaryTensorView>>& arguments,
                     const std::vector<std::shared_ptr<PrimaryTensorView>>& results)
{
    m_tensors.insert(m_tensors.end(), arguments.begin(), arguments.end());
    m_tensors.insert(m_tensors.end(), results.begin(), results.end());
    // TBD
    // From Function allocate tensors for the temporaries
}
