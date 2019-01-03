//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#pragma once

namespace ngraph
{
    namespace op
    {
        class Abs;
        class Acos;
        class Add;
        class Asin;
        class Atan;
        class Ceiling;
        class Cos;
        class Cosh;
        class Exp;
        class Floor;
        class Log;
        class Sin;
        class Sinh;
        class Tan;
        class Tanh;
        class Power;
        class Subtract;
        class Divide;
        class Sign;
        class Maximum;
        class Minimum;
        class Multiply;
        class Convert;
        class Equal;
        class NotEqual;
        class Greater;
        class GreaterEq;
        class Less;
        class LessEq;
        class Not;
        class Relu;
        class ReluBackprop;
        class Max;
        class Min;
        class Negative;
        class Not;
        class Sqrt;
        class Select;
        class And;
        class Or;
        class Nop;
        class Sigmoid;
        class SigmoidBackprop;
    }
    namespace runtime
    {
        namespace gpu
        {
            enum class OpName
            {
                add,
                multiply,
                minimum,
                maximum
            };

            template <typename T>
            struct CudaOpMap;

            template <>
            struct CudaOpMap<ngraph::op::Abs>
            {
                static constexpr const char* op = "fabsf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Acos>
            {
                static constexpr const char* op = "acosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Asin>
            {
                static constexpr const char* op = "asinf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Atan>
            {
                static constexpr const char* op = "atanf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Ceiling>
            {
                static constexpr const char* op = "ceilf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Cos>
            {
                static constexpr const char* op = "cosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Cosh>
            {
                static constexpr const char* op = "coshf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Exp>
            {
                static constexpr const char* op = "expf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Floor>
            {
                static constexpr const char* op = "floorf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Log>
            {
                static constexpr const char* op = "logf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Max>
            {
                static constexpr const char* op = "fmaxf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Min>
            {
                static constexpr const char* op = "fminf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sin>
            {
                static constexpr const char* op = "sinf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sinh>
            {
                static constexpr const char* op = "sinhf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sqrt>
            {
                static constexpr const char* op = "sqrtf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Tan>
            {
                static constexpr const char* op = "tanf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Tanh>
            {
                static constexpr const char* op = "tanhf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Power>
            {
                static constexpr const char* op = "powf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Subtract>
            {
                static constexpr const char* op = "subtractf";
                static constexpr const char* math_kernel = "x0-x1";
                static constexpr const char* atomic = "atomicSub";
            };

            template <>
            struct CudaOpMap<ngraph::op::Divide>
            {
                static constexpr const char* op = "fdividef";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sign>
            {
                static constexpr const char* op = "sign";
                static constexpr const char* math_kernel = "(x0 > 0) - (x0 < 0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::Convert>
            {
                static constexpr const char* op = "convert";
                static constexpr const char* math_kernel = "x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Equal>
            {
                static constexpr const char* op = "equal";
                static constexpr const char* math_kernel = "x0 == x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::NotEqual>
            {
                static constexpr const char* op = "not_equal";
                static constexpr const char* math_kernel = "x0 != x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Greater>
            {
                static constexpr const char* op = "greater";
                static constexpr const char* math_kernel = "x0 > x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::GreaterEq>
            {
                static constexpr const char* op = "greater_equal";
                static constexpr const char* math_kernel = "x0 >= x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Less>
            {
                static constexpr const char* op = "less";
                static constexpr const char* math_kernel = "x0 < x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::LessEq>
            {
                static constexpr const char* op = "less_equal";
                static constexpr const char* math_kernel = "x0 <= x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Relu>
            {
                static constexpr const char* op = "relu";
                static constexpr const char* math_kernel = "fmaxf(0,x0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::Not>
            {
                static constexpr const char* op = "logical_not";
                static constexpr const char* math_kernel = "!x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Negative>
            {
                static constexpr const char* op = "negative";
                static constexpr const char* math_kernel = "-x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Select>
            {
                static constexpr const char* op = "select";
                static constexpr const char* math_kernel = "(x0 == 0) ? x2 : x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::ReluBackprop>
            {
                static constexpr const char* op = "relu_backprop";
                static constexpr const char* math_kernel = "x1 * int(x0 > 0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::And>
            {
                static constexpr const char* op = "logical_and";
                static constexpr const char* math_kernel = "x0 & x1";
                static constexpr const char* atomic = "atomicAnd";
            };

            template <>
            struct CudaOpMap<ngraph::op::Or>
            {
                static constexpr const char* op = "logical_or";
                static constexpr const char* math_kernel = "x0 | x1";
                static constexpr const char* atomic = "atomicOr";
            };

            template <>
            struct CudaOpMap<ngraph::op::Add>
            {
                static constexpr const char* op = "add";
                static constexpr const char* math_kernel = "x0 + x1";
                static constexpr const char* atomic = "atomicAdd";
            };

            template <>
            struct CudaOpMap<ngraph::op::Multiply>
            {
                static constexpr const char* op = "mul";
                static constexpr const char* math_kernel = "x0 * x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Minimum>
            {
                static constexpr const char* op = "min";
                static constexpr const char* math_kernel = "x0 > x1 ? x1 : x0";
                static constexpr const char* atomic = "atomicMin";
            };

            template <>
            struct CudaOpMap<ngraph::op::Maximum>
            {
                static constexpr const char* op = "max";
                static constexpr const char* math_kernel = "x0 > x1 ? x0 : x1";
                static constexpr const char* atomic = "atomicMax";
            };

            template <>
            struct CudaOpMap<ngraph::op::Nop>
            {
                static constexpr const char* op = "";
                static constexpr const char* math_kernel = "";
                static constexpr const char* atomic = "";
            };

            template <>
            struct CudaOpMap<ngraph::op::Sigmoid>
            {
                static constexpr const char* op = "sigmoid";
                static constexpr const char* math_kernel = "1 / (1 + expf(-x0))";
            };

            template <>
            struct CudaOpMap<ngraph::op::SigmoidBackprop>
            {
                static constexpr const char* op = "sigmoid_backprop";
                static constexpr const char* math_kernel = "x1 / (2 + expf(-x0) + expf(x0))";
            };
        }
    }
}
