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

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // acos performs a simple elementwise arccos function.
            template <>
            void Impl<op::Acos>::operator()()
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
            void Impl<op::Asin>::operator()()
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
            void Impl<op::Atan>::operator()()
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
            void Impl<op::Cos>::operator()()
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
            void Impl<op::Cosh>::operator()()
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
            void Impl<op::Exp>::operator()()
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
            void Impl<op::Log>::operator()()
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
            void Impl<op::Power>::operator()()
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
            void Impl<op::Sin>::operator()()
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
            void Impl<op::Sinh>::operator()()
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
            void Impl<op::Sqrt>::operator()()
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
            void Impl<op::Tan>::operator()()
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
            void Impl<op::Tanh>::operator()()
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
                Impl<op::Acos>::Registration register_acos;
                Impl<op::Asin>::Registration register_asin;
                Impl<op::Atan>::Registration register_atan;
                Impl<op::Cos>::Registration register_cos;
                Impl<op::Cosh>::Registration register_cosh;
                Impl<op::Exp>::Registration register_exp;
                Impl<op::Log>::Registration register_log;
                Impl<op::Power>::Registration register_power;
                Impl<op::Sin>::Registration register_sin;
                Impl<op::Sinh>::Registration register_sinh;
                Impl<op::Sqrt>::Registration register_sqrt;
                Impl<op::Tan>::Registration register_tan;
                Impl<op::Tanh>::Registration register_tanh;
            }
        }
    }
}
