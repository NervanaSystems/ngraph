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

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Abs performs a simple elementwise absolute value.
            template <>
            void Impl<op::Abs>::operator()()
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
            void Impl<op::Add>::operator()()
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
            void Impl<op::Ceiling>::operator()()
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
            void Impl<op::Divide>::operator()()
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
            void Impl<op::Floor>::operator()()
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
            void Impl<op::Multiply>::operator()()
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
            void Impl<op::Negative>::operator()()
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
            void Impl<op::Relu>::operator()()
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
            void Impl<op::ReluBackprop>::operator()()
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
            void Impl<op::Sigmoid>::operator()()
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
            void Impl<op::SigmoidBackprop>::operator()()
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
            void Impl<op::Sign>::operator()()
            {
                check_inputs(1);
                check_outputs(1);
                set_output(start_tile_function()
                               .add(builder::Input{op_input(0), "I"})
                               .add(builder::Output{"O"})
                               .add(builder::Elementwise{"S", "(I < 0) ? -1 : ((I > 0) ? 1 : 0)"})
                               .add(builder::Elementwise{
                                   "O", tile_converter("S", op().get_element_type())})
                               .finalize());
            }

            // Subtract performs a simple elementwise subtraction.
            template <>
            void Impl<op::Subtract>::operator()()
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
                Impl<op::Abs>::Registration register_abs;
                Impl<op::Add>::Registration register_add;
                Impl<op::Ceiling>::Registration register_ceiling;
                Impl<op::Divide>::Registration register_divide;
                Impl<op::Floor>::Registration register_floor;
                Impl<op::Multiply>::Registration register_multiply;
                Impl<op::Negative>::Registration register_negative;
                Impl<op::Relu>::Registration register_relu;
                Impl<op::ReluBackprop>::Registration register_relu_backprop;
                Impl<op::Sigmoid>::Registration register_sigmoid;
                Impl<op::SigmoidBackprop>::Registration register_sigmoid_backprop;
                Impl<op::Sign>::Registration register_sign;
                Impl<op::Subtract>::Registration register_subtract;
            }
        }
    }
}
