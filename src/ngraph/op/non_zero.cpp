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

#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/reference/non_zero.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v3::NonZero::type_info;

op::v3::NonZero::NonZero(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::NonZero::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v3::NonZero::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    const auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_integral() || input_et.is_real(),
                          "NonZero input data type needs to be a numeric type. Got: ",
                          input_et);

    set_output_type(0, element::i64, PartialShape{input_shape.rank(), Dimension::dynamic()});
    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v3::NonZero::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v3::NonZero>(new_args.at(0));
}

namespace
{
    template <element::Type_t ET>
    bool try_evaluate_nonzero(Node* node,
                              const EvaluatorTensorPtr& input,
                              const EvaluatorTensorPtr& output)
    {
        if (ET != input->get_element_type() || output->get_element_type() != element::i64 ||
            output->get_element_type() != element::i32)
        {
            return false;
        }

        // input shape should have been resolved to static shape, otherwise, this call assert
        Shape input_shape = input->get_shape();
        size_t input_rank = input_shape.size();

        size_t non_zero_count =
            runtime::reference::non_zero_get_count(input->get_ptr<ET>(), input_shape);

        // ??? At this point, we know the output size should be,
        // i.e., shape {rank(input_shape), non_zero_count} or {1} or {0}
        // BUT what to do with this info?
        // 1) How to set the output tensor size? OR what is the current size of output?
        // 2) What to do with the case of 0 non-zero items?
        // Below is the guess on what to do with Scott's PR 4610
        Shape out_shape;
        if (non_zero_count == 0)
        {
            out_shape = Shape{0};
        }
        else if (input_rank == 0)
        {
            out_shape = Shape{1, 1};
        }
        else
        {
            out_shape = Shape{input_rank, non_zero_count};
        }

        //    output->set_shape(node, out_shape);
        runtime::reference::non_zero(
            input->get_ptr<ET>(), output->get_ptr<element::Type_t::i64>(), out_shape);

        return true;
    }
}

bool op::v3::NonZero::evaluate(const EvaluatorTensorVector& outputs,
                               const EvaluatorTensorVector& inputs)
{
    return try_evaluate_nonzero<element::Type_t::i8>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::i16>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::i32>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::i64>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::u8>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::u16>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::u32>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::u64>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::bf16>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::f32>(this, inputs[0], outputs[0]) ||
           try_evaluate_nonzero<element::Type_t::f64>(this, inputs[0], outputs[0]);
}
