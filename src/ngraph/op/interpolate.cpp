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

#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Interpolate::type_info;

op::Interpolate::Interpolate(const Output<Node>& image,
                             const Output<Node>& output_shape,
                             const InterpolateAttrs& attrs)
    : Op({image, output_shape})
    , m_attrs(attrs)
{
    m_attrs.mode = as_string<InterpolateMode>(as_enum<InterpolateMode>(attrs.mode));
    constructor_validate_and_infer_types();
}

bool op::Interpolate::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs.axes", m_attrs.axes);
    visitor.on_attribute("attrs.mode", m_attrs.mode);
    visitor.on_attribute("attrs.align_corners", m_attrs.align_corners);
    visitor.on_attribute("attrs.antialias", m_attrs.antialias);
    visitor.on_attribute("attrs.pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("attrs.pads_end", m_attrs.pads_end);
    return true;
}

void op::Interpolate::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "output shape must be an integral number.");
    set_input_is_relevant_to_shape(1);

    PartialShape output_shape = PartialShape(get_input_partial_shape(0));
    if (output_shape.rank().is_static())
    {
        for (auto axis : m_attrs.axes)
        {
            NGRAPH_CHECK(axis < output_shape.rank().get_length());
            output_shape[axis] = Dimension::dynamic();
        }
    }

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        auto out_shape = const_shape->cast_vector<int64_t>();
        size_t i = 0;
        for (auto axis : m_attrs.axes)
        {
            output_shape[axis] = Dimension(out_shape[i++]);
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
}

namespace ngraph
{
    template <>
    EnumNames<op::Interpolate::InterpolateMode>& EnumNames<op::Interpolate::InterpolateMode>::get()
    {
        static auto enum_names = EnumNames<op::Interpolate::InterpolateMode>(
            "op::Interpolate::InterpolateMode",
            {{"nearest", op::Interpolate::InterpolateMode::NEAREST},
             {"linear", op::Interpolate::InterpolateMode::LINEAR},
             {"cubic", op::Interpolate::InterpolateMode::CUBIC},
             {"area", op::Interpolate::InterpolateMode::AREA}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::Interpolate::InterpolateMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::Interpolate::InterpolateMode& type)
    {
        return s << as_string(type);
    }
}

// Interpolate v3

constexpr NodeTypeInfo op::v3::Interpolate::type_info;

op::v3::Interpolate::Interpolate(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const op::v3::InterpolateAttrs& attrs)
    : Op({image, output_shape})
    , m_attrs(attrs)
{
    m_attrs.mode = as_string<InterpolateMode>(as_enum<InterpolateMode>(attrs.mode));
    m_attrs.coordinate_transformation_mode = as_string<CoordinateTransformMode>(
        as_enum<CoordinateTransformMode>(attrs.coordinate_transformation_mode));
    constructor_validate_and_infer_types();
}

bool op::v3::Interpolate::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs.axes", m_attrs.axes);
    visitor.on_attribute("attrs.mode", m_attrs.mode);
    visitor.on_attribute("attrs.coordinate_transformation_mode",
                         m_attrs.coordinate_transformation_mode);
    visitor.on_attribute("attrs.align_corners", m_attrs.align_corners);
    visitor.on_attribute("attrs.antialias", m_attrs.antialias);
    visitor.on_attribute("attrs.pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("attrs.pads_end", m_attrs.pads_end);
    return true;
}

void op::v3::Interpolate::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "output shape must be an integral number.");
    set_input_is_relevant_to_shape(1);

    PartialShape output_shape = PartialShape(get_input_partial_shape(0));
    if (output_shape.rank().is_static())
    {
        for (auto axis : m_attrs.axes)
        {
            NGRAPH_CHECK(axis < output_shape.rank().get_length());
            output_shape[axis] = Dimension::dynamic();
        }
    }

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        auto out_shape = const_shape->cast_vector<int64_t>();
        size_t i = 0;
        for (auto axis : m_attrs.axes)
        {
            output_shape[axis] = Dimension(out_shape[i++]);
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v3::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v3::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
}

namespace ngraph
{
    template <>
    EnumNames<op::v3::Interpolate::InterpolateMode>&
        EnumNames<op::v3::Interpolate::InterpolateMode>::get()
    {
        static auto enum_names = EnumNames<op::v3::Interpolate::InterpolateMode>(
            "op::v3::Interpolate::InterpolateMode",
            {{"nearest", op::v3::Interpolate::InterpolateMode::NEAREST},
             {"linear", op::v3::Interpolate::InterpolateMode::LINEAR},
             {"cubic", op::v3::Interpolate::InterpolateMode::CUBIC},
             {"area", op::v3::Interpolate::InterpolateMode::AREA}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v3::Interpolate::InterpolateMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v3::Interpolate::InterpolateMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::v3::Interpolate::CoordinateTransformMode>&
        EnumNames<op::v3::Interpolate::CoordinateTransformMode>::get()
    {
        static auto enum_names = EnumNames<op::v3::Interpolate::CoordinateTransformMode>(
            "op::v3::Interpolate::CoordinateTransformMode",
            {{"half_pixel", op::v3::Interpolate::CoordinateTransformMode::HALF_PIXEL},
             {"pytorch_half_pixel",
              op::v3::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL},
             {"asymmetric", op::v3::Interpolate::CoordinateTransformMode::ASYMMETRIC},
             {"tf_half_pixel_for_nn",
              op::v3::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v3::Interpolate::CoordinateTransformMode>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v3::Interpolate::CoordinateTransformMode& type)
    {
        return s << as_string(type);
    }
}
