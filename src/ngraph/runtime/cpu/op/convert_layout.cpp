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

#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::op::ConvertLayout::ConvertLayout(
    const shared_ptr<Node>& arg, const shared_ptr<runtime::cpu::LayoutDescriptor>& layout)
    : ConvertLayout(arg, 0, layout)
{
}

shared_ptr<Node>
    runtime::cpu::op::ConvertLayout::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ConvertLayout>(new_args.at(0), output_layout);
}

runtime::cpu::op::ConvertLayout::ConvertLayout(
    const shared_ptr<Node>& arg,
    size_t output_index,
    const shared_ptr<runtime::cpu::LayoutDescriptor>& layout)
    : Op("ConvertLayout", check_single_output_args({arg}))
    , arg_output_index(output_index)
    , output_layout(layout)
{
    constructor_validate_and_infer_types();
}

void runtime::cpu::op::ConvertLayout::validate_and_infer_types()
{
    const auto& arg = get_argument(0);

    const auto& arg_tensor = arg->get_output_tensor_ptr(arg_output_index);
    const auto& arg_layout = arg_tensor->get_tensor_layout();

    if (!arg_layout)
    {
        // throw ngraph_error("Layout conversion input tensor is missing layout information");
    }

    set_output_type(0, output_layout->get_element_type(), output_layout->get_shape());
    get_output_tensor_ptr()->set_tensor_layout(output_layout);
}
