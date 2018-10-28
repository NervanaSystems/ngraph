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

#include "ngraph/op/convert.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

// Convert views a tensor as a new type.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Convert>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{
                       "O", tile_converter("I", to_plaidml(op().get_convert_element_type()))})
                   .finalize());
}

namespace
{
    ngraph::runtime::plaidml::Impl<ngraph::op::Convert>::Registration register_convert;
}
