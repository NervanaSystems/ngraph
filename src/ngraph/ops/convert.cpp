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

#include <memory>

#include "ngraph/ops/convert.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::op;

const element::Type& Convert::propagate_element_types(const element::Type& arg_element_type) const
{
    return m_element_type;
}

void ngraph::op::Convert::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto x = m_arguments[0];

    adjoints.add_delta(x, std::make_shared<op::Convert>(delta, m_element_type));
}
