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

#include "ngraph/op/ceiling.hpp"

using namespace std;
using namespace ngraph;

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"

constexpr NodeTypeInfo op::Ceiling::type_info;

op::Ceiling::Ceiling(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Ceiling::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Ceiling>(new_args.at(0));
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ceiling<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    template <element::Type_t ET>
    inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        memcpy(out->get_data_ptr<T>(), arg0->get_data_ptr<T>(), count * sizeof(T));
        return true;
    }

    bool evaluate_ceiling(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

#define IDENTITY(a)                                                                                \
    case element::Type_t::a: rc = copy_tensor<element::Type_t::a>

        switch (arg0->get_element_type())
        {
            IDENTITY(boolean)(arg0, out, count);
            break;
            IDENTITY(i8)(arg0, out, count);
            break;
            IDENTITY(i16)(arg0, out, count);
            break;
            IDENTITY(i32)(arg0, out, count);
            break;
            IDENTITY(i64)(arg0, out, count);
            break;
            IDENTITY(u8)(arg0, out, count);
            break;
            IDENTITY(u16)(arg0, out, count);
            break;
            IDENTITY(u32)(arg0, out, count);
            break;
            IDENTITY(u64)(arg0, out, count);
            break;
            TYPE_CASE(bf16)(arg0, out, count);
            break;
            TYPE_CASE(f16)(arg0, out, count);
            break;
            TYPE_CASE(f32)(arg0, out, count);
            break;
            TYPE_CASE(f64)(arg0, out, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Ceiling::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_ceiling(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
