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

#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/pass/constant_folding.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ShapeOf::type_info;

op::ShapeOf::ShapeOf(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

void op::ShapeOf::validate_and_infer_types()
{
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, PartialShape{get_input_partial_shape(0).rank()});
}

bool ngraph::op::v0::ShapeOf::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto new_shape_of = make_shared<ShapeOf>(new_args.at(0));
    new_shape_of->set_is_foldable(m_is_foldable);
    return new_shape_of;
}

OutputVector op::ShapeOf::constant_fold_default()
{
    auto arg_match = input_value(0);
    auto partial_shape = arg_match.get_partial_shape();
    OutputVector replacements;
    if (partial_shape.is_static())
    {
        NGRAPH_CHECK(pass::revalidate_and_ensure_static(shared_from_this()));
        auto arg_shape = arg_match.get_shape();
        replacements = OutputVector{
            make_shared<op::Constant>(element::i64, Shape{arg_shape.size()}, arg_shape.data())};
    }
    else if (partial_shape.rank().is_static() && m_is_foldable)
    {
        auto shape_of = make_shared<op::ShapeOf>(arg_match);
        shape_of->set_is_foldable(false);
        auto dimensions = OutputVector{};
        auto output_dimensions = vector<Dimension>(partial_shape);
        for (size_t i = 0; i < output_dimensions.size(); ++i)
        {
            if (output_dimensions[i].is_static())
            {
                auto temp =
                    op::Constant::create(element::i64,
                                         Shape{1},
                                         {static_cast<int64_t>(output_dimensions[i].get_length())});
                temp->set_friendly_name("ConstDim/" + temp->get_name());
                dimensions.push_back(temp);
            }
            else
            {
                auto index = op::Constant::create(element::i64, Shape{1}, {i});
                auto axis = op::Constant::create(element::i64, Shape{}, {0});
                auto temp = make_shared<op::v1::Gather>(shape_of, index, axis);
                temp->set_friendly_name("DynDim/" + temp->get_name());
                dimensions.push_back(temp);
            }
        }

        replacements = OutputVector{std::make_shared<op::Concat>(dimensions, 0)};
    }
    return replacements;
}
