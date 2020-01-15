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

#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplAbs, OpImpl<op::Abs>);
            NGRAPH_PLAIDML_OP_CLASS(ImplAdd, OpImpl<op::Add>);
            NGRAPH_PLAIDML_OP_CLASS(ImplCeiling, OpImpl<op::Ceiling>);
            NGRAPH_PLAIDML_OP_CLASS(ImplDivide, OpImpl<op::Divide>);
            NGRAPH_PLAIDML_OP_CLASS(ImplFloor, OpImpl<op::Floor>);
            NGRAPH_PLAIDML_OP_CLASS(ImplMultiply, OpImpl<op::Multiply>);
            NGRAPH_PLAIDML_OP_CLASS(ImplNegative, OpImpl<op::Negative>);
            NGRAPH_PLAIDML_OP_CLASS(ImplRelu, OpImpl<op::Relu>);
            NGRAPH_PLAIDML_OP_CLASS(ImplReluBackprop, OpImpl<op::ReluBackprop>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSigmoid, OpImpl<op::Sigmoid>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSigmoidBackprop, OpImpl<op::SigmoidBackprop>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSign, OpImpl<op::Sign>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSubtract, OpImpl<op::Subtract>);
        }
    }
}

// Abs performs a simple elementwise absolute value.
void ngraph::runtime::plaidml::ImplAbs::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "abs(I)"})
                   .finalize());
}

// Add performs a simple elementwise addition.
void ngraph::runtime::plaidml::ImplAdd::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A + B"})
                   .finalize());
}

// Ceiling performs a simple elementwise ceiling.
void ngraph::runtime::plaidml::ImplCeiling::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "ceil(I)"})
                   .finalize());
}

// Divide performs a simple elementwise division.
void ngraph::runtime::plaidml::ImplDivide::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A / B"})
                   .finalize());
}

// Floor performs a simple elementwise floor.
void ngraph::runtime::plaidml::ImplFloor::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "floor(I)"})
                   .finalize());
}

// Multiply performs a simple elementwise multiplication.
void ngraph::runtime::plaidml::ImplMultiply::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A * B"})
                   .finalize());
}

// Negative performs a simple elementwise negation.
void ngraph::runtime::plaidml::ImplNegative::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "-I"})
                   .finalize());
}

// Relu implements a simple elementwise rectified linear unit.
void ngraph::runtime::plaidml::ImplRelu::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "relu(I)"})
                   .finalize());
}

// ReluBackprop computes the derivative of Relu.
void ngraph::runtime::plaidml::ImplReluBackprop::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Input{op_input(1), "DO"})
                   .add(builder::Output{"DI"})
                   .add(builder::Elementwise{"DI", "I > 0 ? DO : 0"})
                   .finalize());
}

// Sigmoid computes a standard ML sigmoid: 1/(1+exp(-X))
void ngraph::runtime::plaidml::ImplSigmoid::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "1/(1+exp(-I))"})
                   .finalize());
}

// SigmoidBackprop computes the derivative of a standard ML
// sigmoid: dOutput * sigmoid(X) * (1-sigmoid(X))
void ngraph::runtime::plaidml::ImplSigmoidBackprop::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Input{op_input(1), "DO"})
                   .add(builder::Output{"DI"})
                   .add(builder::Elementwise{"O", "1/(1+exp(-I))"})
                   .add(builder::Elementwise{"DI", "DO * O * (1-O)"})
                   .finalize());
}

// Sign returns the sign of an element.
void ngraph::runtime::plaidml::ImplSign::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"S", "(I < 0) ? -1 : ((I > 0) ? 1 : 0)"})
                   .add(builder::Elementwise{"O", tile_converter("S", op().get_element_type())})
                   .finalize());
}

// Subtract performs a simple elementwise subtraction.
void ngraph::runtime::plaidml::ImplSubtract::Apply()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A - B"})
                   .finalize());
}
