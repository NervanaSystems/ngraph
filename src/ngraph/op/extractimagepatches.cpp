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

#include "ngraph/op/extractimagepatches.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ExtractImagePatches::type_info;


// ExtractImagePatches v3

constexpr NodeTypeInfo op::v3::ExtractImagePatches::type_info;

op::v3::ExtractImagePatches::ExtractImagePatches(const Output<Node>& image,
                                 const op::v3::ExtractImagePatchesAttrs& attrs)
    : Op({image, output_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::v3::ExtractImagePatches::validate_and_infer_types()
{
    Shape input_shape = get_input_shape(0);
    
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number(),
                          "input tensor must be an integral number.");
    NODE_VALIDATION_CHECK(this,
                          input_shape.size()==4,
                          "input tensor must be 4D tensor.");

    NODE_VALIDATION_CHECK(this,
                          m_attrs.patch_sizes.size()==2,
                          "Attribute sizes should be in [size_rows, size_cols] format.");
    
    NODE_VALIDATION_CHECK(this,
                          m_attrs.patch_movement_strides.size()==2,
                          "Attribute strides should be in [stride_rows, stride_cols] format.");
    
    
    NODE_VALIDATION_CHECK(this,
                          m_attrs.patch_selection_rates.size()==2,
                          "Attribute rates should be in [rate_rows, rate_cols] format.");
    
    NODE_VALIDATION_CHECK(this,
                          m_attrs.padding==PadType::VALID || m_attrs.padding==PadType::SAME_LOWER ||  m_attrs.padding==PadType::SAME_UPPER,
                          "Attribute padding should be in either valid or same_lower or same_upper.");
    
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1); //output shape also depends on attribute 

    size_t out_rows((input_shape[2]-1)/m_attrs.patch_movement_strides[0]);
    size_t out_cols((input_shape[3]-1)/m_attrs.patch_movement_strides[1]);
    if(m_attrs.padding == PadType::VALID){
        out_rows  =  ((input_shape[2]-(m_attrs.patch_selection_rates[0])*(m_attrs.patch_sizes[0]-1)-1)/m_attrs.patch_movement_strides[0])+1;
        out_cols  =  ((input_shape[3]-(m_attrs.patch_selection_rates[1])*(m_attrs.patch_sizes[1]-1)-1)/m_attrs.patch_movement_strides[1])+1;
    }
    Shape output_shape;
    output_shape.push_back(input_shape[0]);
    output_shape.push_back( input_shape[1]*patch_sizes[0]+patch_sizes[1]); // size[1]*size[2]*depth
    output_shape.push_back( out_rows ); 
    output_shape.push_back( out_cols );

    if(input_shape[2]==0 || input_shape[3]==0){
        output_shape=input_shape;
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

bool op::v3::ExtractImagePatches::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("sizes", m_attrs.patch_sizes);
    visitor.on_attribute("strides", m_attrs.patch_movement_strides);
    visitor.on_attribute("rates", m_attrs.rates);
    visitor.on_attribute("padding", m_attrs.padding);
    return true;
}

shared_ptr<Node> op::v3::ExtractImagePatches::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v3::ExtractImagePatches>(new_args.at(0), m_attrs);
}

