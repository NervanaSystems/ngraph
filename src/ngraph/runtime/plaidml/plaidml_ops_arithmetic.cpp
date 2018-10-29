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

// Abs performs a simple elementwise absolute value.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Abs>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Add>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Ceiling>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Divide>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Floor>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Multiply>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Negative>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Relu>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::ReluBackprop>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Sigmoid>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::SigmoidBackprop>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Sign>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Subtract>::operator()()
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

namespace
{
    ngraph::runtime::plaidml::Impl<ngraph::op::Abs>::Registration register_abs;
    ngraph::runtime::plaidml::Impl<ngraph::op::Add>::Registration register_add;
    ngraph::runtime::plaidml::Impl<ngraph::op::Ceiling>::Registration register_ceiling;
    ngraph::runtime::plaidml::Impl<ngraph::op::Divide>::Registration register_divide;
    ngraph::runtime::plaidml::Impl<ngraph::op::Floor>::Registration register_floor;
    ngraph::runtime::plaidml::Impl<ngraph::op::Multiply>::Registration register_multiply;
    ngraph::runtime::plaidml::Impl<ngraph::op::Negative>::Registration register_negative;
    ngraph::runtime::plaidml::Impl<ngraph::op::Relu>::Registration register_relu;
    ngraph::runtime::plaidml::Impl<ngraph::op::ReluBackprop>::Registration register_relu_backprop;
    ngraph::runtime::plaidml::Impl<ngraph::op::Sigmoid>::Registration register_sigmoid;
    ngraph::runtime::plaidml::Impl<ngraph::op::SigmoidBackprop>::Registration
        register_sigmoid_backprop;
    ngraph::runtime::plaidml::Impl<ngraph::op::Sign>::Registration register_sign;
    ngraph::runtime::plaidml::Impl<ngraph::op::Subtract>::Registration register_subtract;
}
