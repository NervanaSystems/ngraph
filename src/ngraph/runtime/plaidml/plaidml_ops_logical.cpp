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

#include "ngraph/op/and.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplAnd, OpImpl<op::And>);
            NGRAPH_PLAIDML_OP_CLASS(ImplNot, OpImpl<op::Not>);
            NGRAPH_PLAIDML_OP_CLASS(ImplOr, OpImpl<op::Or>);
            NGRAPH_PLAIDML_OP_CLASS(ImplXor, OpImpl<op::Xor>);
        }
    }
}

// And performs a simple elementwise logical and.
void ngraph::runtime::plaidml::ImplAnd::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A ? B : A"})
                   .finalize());
}

// Not performs a simple elementwise logical not.
void ngraph::runtime::plaidml::ImplNot::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "cmp_eq(I, 0)"})
                   .finalize());
}

// Or performs a simple elementwise logical or.
void ngraph::runtime::plaidml::ImplOr::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A ? A : B"})
                   .finalize());
}

// Xor performs a simple elementwise logical xor.
void ngraph::runtime::plaidml::ImplXor::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A ? (B ? 0 : A) : B"})
                   .finalize());
}
