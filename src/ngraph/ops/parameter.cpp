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

#include <sstream>

#include "ngraph/ops/parameter.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph::op;

Parameter::Parameter(const std::shared_ptr<const ValueType>& value_type)
    : Node(value_type)
{
}

Parameter::Parameter(const ngraph::element::Type& element_type, const Shape& shape)
    : Parameter(make_shared<TensorViewType>(element_type, shape))
{
}

void Parameter::propagate_types()
{
}

void Parameter::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
}
