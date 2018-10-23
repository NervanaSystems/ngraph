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

#include "ngraph/op/equal.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

// Equal performs a simple elementwise equality.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Equal>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0, TensorContents::LOGICAL), "A"})
                   .add(builder::Input{op_input(1, TensorContents::LOGICAL), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A == B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

// Greater performs a simple elementwise greater-than comparison.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Greater>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A > B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

// GreaterEq performs a simple elementwise greater-than-or-equal-to comparison.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::GreaterEq>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A >= B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

// Less performs a simple elementwise less-than comparison.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Less>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A < B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

// LessEq performs a simple elementwise less-than-or-equal-to comparison.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::LessEq>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0), "A"})
                   .add(builder::Input{op_input(1), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A <= B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

// Maximum performs a simple elementwise maximum.
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Maximum>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::Minimum>::operator()()
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
template <>
void ngraph::runtime::plaidml::Impl<ngraph::op::NotEqual>::operator()()
{
    check_inputs(2);
    check_outputs(1);
    set_output(start_tile_function()
                   .add(builder::Input{op_input(0, TensorContents::LOGICAL), "A"})
                   .add(builder::Input{op_input(1, TensorContents::LOGICAL), "B"})
                   .add(builder::Output{"C"})
                   .add(builder::Elementwise{"C", "A != B"})
                   .finalize(),
               TensorContents::LOGICAL);
}

namespace
{
    ngraph::runtime::plaidml::Impl<ngraph::op::Equal>::Registration register_equal;
    ngraph::runtime::plaidml::Impl<ngraph::op::Greater>::Registration register_greater;
    ngraph::runtime::plaidml::Impl<ngraph::op::GreaterEq>::Registration register_greater_eq;
    ngraph::runtime::plaidml::Impl<ngraph::op::Less>::Registration register_less;
    ngraph::runtime::plaidml::Impl<ngraph::op::LessEq>::Registration register_less_eq;
    ngraph::runtime::plaidml::Impl<ngraph::op::Maximum>::Registration register_maximum;
    ngraph::runtime::plaidml::Impl<ngraph::op::Minimum>::Registration register_minimum;
    ngraph::runtime::plaidml::Impl<ngraph::op::NotEqual>::Registration register_not_equal;
}
