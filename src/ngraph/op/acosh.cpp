//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/acosh.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"

#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::Acosh::type_info;

op::v3::Acosh::Acosh(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::Acosh::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Acosh>(new_args.at(0));
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::acosh(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg0->get_shape(), broadcast_spec);
        return true;
    }
}

bool op::v3::Acosh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    const HostTensorPtr& arg0 = inputs[0];
    const HostTensorPtr& out = outputs[0];
    const op::AutoBroadcastSpec& broadcast_spec = get_autob();
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0);
    switch (arg0->get_element_type())
    {
        TYPE_CASE(i8)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(i16)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(i32)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(i64)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(u8)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(u16)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(u32)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(u64)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(f32)(arg0, out, broadcast_spec);
        break;
        TYPE_CASE(f64)(arg0, out, broadcast_spec);
        break;
    default: rc = false; break;
    }
    return rc;
}
