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
#include <vector>

#include "ngraph/ops/xla_tuple.hpp"

using namespace std;
using namespace ngraph;

op::XLATuple::XLATuple(const Nodes& args)
    : Node("XLATuple", args)
{
    vector<shared_ptr<const ValueType>> element_types;
    for (auto argument : get_arguments_via_inputs())
    {
        element_types.push_back(argument->get_value_type());
    }
    set_value_type_checked(make_shared<TupleType>(element_types));
}
