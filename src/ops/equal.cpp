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
using namespace ngraph::op;

const element::Type& Equal::infer_result_element_type(const element::Type& arg0_element_type,
                                                      const element::Type& arg1_element_type) const
{
    if (arg0_element_type != arg1_element_type)
    {
        throw ngraph_error("Arguments must have the same tensor view element type");
    }

    return m_result_element_type;
}
