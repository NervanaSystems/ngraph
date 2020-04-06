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

#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::BinaryConvolution::type_info;

op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                             const Output<Node>& kernel,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const BinaryConvolutionMode& mode,
                                             float pad_value,
                                             const PadType& auto_pad)
    : Op({data, kernel})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_mode(mode)
    , m_pad_value(pad_value)
    , m_auto_pad(auto_pad)
{
    constructor_validate_and_infer_types();
}

op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                             const Output<Node>& kernel,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const std::string& mode,
                                             float pad_value,
                                             const PadType& auto_pad)
    : BinaryConvolution(data,
                        kernel,
                        strides,
                        pads_begin,
                        pads_end,
                        dilations,
                        mode_from_string(mode),
                        pad_value,
                        auto_pad)
{
}

void op::v1::BinaryConvolution::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    if (m_strides.size() == 0)
    {
        m_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_dilations.size() == 0)
    {
        m_dilations = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_pads_begin.size() == 0)
    {
        m_pads_begin = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_pads_end.size() == 0)
    {
        m_pads_end = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        if (data_batch_shape.is_static() && filters_shape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
        }
    }

    PartialShape result_shape;
    result_shape = infer_convolution_forward(this,
                                             data_batch_shape,
                                             Strides(data_batch_shape.rank().get_length() - 2, 1),
                                             m_pads_begin,
                                             m_pads_end,
                                             filters_shape,
                                             m_strides,
                                             m_dilations);

    set_output_type(0, data_batch_et, result_shape);
}

shared_ptr<Node>
    op::v1::BinaryConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::BinaryConvolution>(new_args.at(0),
                                              new_args.at(1),
                                              m_strides,
                                              m_pads_begin,
                                              m_pads_end,
                                              m_dilations,
                                              m_mode,
                                              m_pad_value,
                                              m_auto_pad);
}

bool op::v1::BinaryConvolution::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("pad_value", m_pad_value);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::BinaryConvolution::generate_adjoints(autodiff::Adjoints& adjoints,
                                                  const OutputVector& deltas)
{
    throw ngraph_error("BinaryConvolution generate_adjoints not implemented");
}

namespace ngraph
{
    template <>
    EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>&
        EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>::get()
    {
        static auto enum_names = EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>(
            "op::v1::BinaryConvolution::BinaryConvolutionMode",
            {{"xnor-popcount", op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v1::BinaryConvolution::BinaryConvolutionMode& type)
    {
        return s << as_string(type);
    }
}

op::v1::BinaryConvolution::BinaryConvolutionMode
    op::v1::BinaryConvolution::mode_from_string(const std::string& mode) const
{
    return as_enum<BinaryConvolutionMode>(mode);
}
