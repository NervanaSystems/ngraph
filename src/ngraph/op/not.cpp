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

#include "ngraph/op/not.hpp"
#include "ngraph/op/op.hpp"

using namespace ngraph;
using namespace std;

op::Not::Not(const shared_ptr<Node>& arg)
    : Op("Not", check_single_output_args({arg}))
{
    constructor_validate_and_infer_types();
}

// TODO(amprocte): Update this to allow only boolean, for consistency with logical binops.
void op::Not::validate_and_infer_types()
{
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type(0, args_et, args_pshape);
}

shared_ptr<Node> op::Not::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Not>(new_args.at(0));
}
