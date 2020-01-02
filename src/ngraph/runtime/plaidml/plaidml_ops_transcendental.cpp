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

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplAcos, OpImpl<op::Acos>);
            NGRAPH_PLAIDML_OP_CLASS(ImplAsin, OpImpl<op::Asin>);
            NGRAPH_PLAIDML_OP_CLASS(ImplAtan, OpImpl<op::Atan>);
            NGRAPH_PLAIDML_OP_CLASS(ImplCos, OpImpl<op::Cos>);
            NGRAPH_PLAIDML_OP_CLASS(ImplCosh, OpImpl<op::Cosh>);
            NGRAPH_PLAIDML_OP_CLASS(ImplExp, OpImpl<op::Exp>);
            NGRAPH_PLAIDML_OP_CLASS(ImplLog, OpImpl<op::Log>);
            NGRAPH_PLAIDML_OP_CLASS(ImplPower, OpImpl<op::Power>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSin, OpImpl<op::Sin>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSinh, OpImpl<op::Sinh>);
            NGRAPH_PLAIDML_OP_CLASS(ImplSqrt, OpImpl<op::Sqrt>);
            NGRAPH_PLAIDML_OP_CLASS(ImplTan, OpImpl<op::Tan>);
            NGRAPH_PLAIDML_OP_CLASS(ImplTanh, OpImpl<op::Tanh>);
        }
    }
}

// acos performs a simple elementwise arccos function.
void ngraph::runtime::plaidml::ImplAcos::Apply()
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
void ngraph::runtime::plaidml::ImplAsin::Apply()
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
void ngraph::runtime::plaidml::ImplAtan::Apply()
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
void ngraph::runtime::plaidml::ImplCos::Apply()
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
void ngraph::runtime::plaidml::ImplCosh::Apply()
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
void ngraph::runtime::plaidml::ImplExp::Apply()
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
void ngraph::runtime::plaidml::ImplLog::Apply()
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
void ngraph::runtime::plaidml::ImplPower::Apply()
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
void ngraph::runtime::plaidml::ImplSin::Apply()
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
void ngraph::runtime::plaidml::ImplSinh::Apply()
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
void ngraph::runtime::plaidml::ImplSqrt::Apply()
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
void ngraph::runtime::plaidml::ImplTan::Apply()
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
void ngraph::runtime::plaidml::ImplTanh::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "I"})
                   .add(builder::Output{"O"})
                   .add(builder::Elementwise{"O", "tanh(I)"})
                   .finalize());
}
