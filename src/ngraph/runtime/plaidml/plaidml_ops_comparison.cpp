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

#include "ngraph/op/equal.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplEqual, OpImpl<op::Equal>);
            NGRAPH_PLAIDML_OP_CLASS(ImplGreater, OpImpl<op::Greater>);
            NGRAPH_PLAIDML_OP_CLASS(ImplGreaterEq, OpImpl<op::GreaterEq>);
            NGRAPH_PLAIDML_OP_CLASS(ImplLess, OpImpl<op::Less>);
            NGRAPH_PLAIDML_OP_CLASS(ImplLessEq, OpImpl<op::LessEq>);
            NGRAPH_PLAIDML_OP_CLASS(ImplMaximum, OpImpl<op::Maximum>);
            NGRAPH_PLAIDML_OP_CLASS(ImplMinimum, OpImpl<op::Minimum>);
            NGRAPH_PLAIDML_OP_CLASS(ImplNotEqual, OpImpl<op::NotEqual>);
        }
    }
}

// Equal performs a simple elementwise equality.
void ngraph::runtime::plaidml::ImplEqual::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A == B"})
                   .finalize());
}

// Greater performs a simple elementwise greater-than comparison.
void ngraph::runtime::plaidml::ImplGreater::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A > B"})
                   .finalize());
}

// GreaterEq performs a simple elementwise greater-than-or-equal-to comparison.
void ngraph::runtime::plaidml::ImplGreaterEq::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A >= B"})
                   .finalize());
}

// Less performs a simple elementwise less-than comparison.
void ngraph::runtime::plaidml::ImplLess::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A < B"})
                   .finalize());
}

// LessEq performs a simple elementwise less-than-or-equal-to comparison.
void ngraph::runtime::plaidml::ImplLessEq::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A <= B"})
                   .finalize());
}

// Maximum performs a simple elementwise maximum.
void ngraph::runtime::plaidml::ImplMaximum::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "max(A, B)"})
                   .finalize());
}

// Minimum performs a simple elementwise minimum.
void ngraph::runtime::plaidml::ImplMinimum::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "min(A, B)"})
                   .finalize());
}

// NotEqual performs a simple elementwise not-equality.
void ngraph::runtime::plaidml::ImplNotEqual::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A != B"})
                   .finalize());
}
