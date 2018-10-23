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

#include "ngraph/op/acos.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

// acos performs a simple elementwise arccos function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Acos>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "acos(I)"})
                   .finalize());
}

// asin performs a simple elementwise arcsin function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Asin>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "asin(I)"})
                   .finalize());
}

// atan performs a simple elementwise arctan function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Atan>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "atan(I)"})
                   .finalize());
}

// cos performs a simple elementwise cos function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Cos>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "cos(I)"})
                   .finalize());
}

// cosh performs a simple elementwise hyperbolic cos function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Cosh>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "cosh(I)"})
                   .finalize());
}

// exp performs a simple elementwise natural exponential function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Exp>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "exp(I)"})
                   .finalize());
}

// log performs a simple elementwise natural logarithm function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Log>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "log(I)"})
                   .finalize());
}

// power performs a simple elementwise power function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Power>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Input{op_input(1), "E"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "pow(I, E)"})
                   .finalize());
}

// sin performs a simple elementwise sin function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Sin>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "sin(I)"})
                   .finalize());
}

// sinh performs a simple elementwise hyperbolic sin function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Sinh>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "sinh(I)"})
                   .finalize());
}

// sqrt performs a simple elementwise square root function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Sqrt>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "sqrt(I)"})
                   .finalize());
}

// tan performs a simple elementwise tangent function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Tan>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "tan(I)"})
                   .finalize());
}

// tanh performs a simple elementwise hyperbolic tangent function.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Tanh>::operator()()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "tanh(I)"})
                   .finalize());
}

namespace
{
    ngraph::runtime::plaidml::Impl<ngraph::op::Acos>::Registration register_acos;
    ngraph::runtime::plaidml::Impl<ngraph::op::Asin>::Registration register_asin;
    ngraph::runtime::plaidml::Impl<ngraph::op::Atan>::Registration register_atan;
    ngraph::runtime::plaidml::Impl<ngraph::op::Cos>::Registration register_cos;
    ngraph::runtime::plaidml::Impl<ngraph::op::Cosh>::Registration register_cosh;
    ngraph::runtime::plaidml::Impl<ngraph::op::Exp>::Registration register_exp;
    ngraph::runtime::plaidml::Impl<ngraph::op::Log>::Registration register_log;
    ngraph::runtime::plaidml::Impl<ngraph::op::Power>::Registration register_power;
    ngraph::runtime::plaidml::Impl<ngraph::op::Sin>::Registration register_sin;
    ngraph::runtime::plaidml::Impl<ngraph::op::Sinh>::Registration register_sinh;
    ngraph::runtime::plaidml::Impl<ngraph::op::Sqrt>::Registration register_sqrt;
    ngraph::runtime::plaidml::Impl<ngraph::op::Tan>::Registration register_tan;
    ngraph::runtime::plaidml::Impl<ngraph::op::Tanh>::Registration register_tanh;
}
