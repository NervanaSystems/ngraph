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

#include "ngraph/runtime/plaidml/plaidml_translate.hpp"
#include "ngraph/runtime/plaidml/plaidml_builder.hpp"

namespace vp = vertexai::plaidml;

vp::datatype ngraph::runtime::plaidml::to_plaidml(const ngraph::element::Type& element_type,
                                                  ConversionUse use)
{
    switch (element_type.bitwidth())
    {
    case 8:
        if (element_type.is_signed())
        {
            // TODO: Extend nGraph's element::Type to distinguish between boolean and i8.
            if (element_type.c_type_string() == "char" && use == ConversionUse::FOR_DATA)
            {
                return PLAIDML_DATA_BOOLEAN;
            }
            return PLAIDML_DATA_INT8;
        }
        return PLAIDML_DATA_UINT8;

    case 16:
        if (element_type.is_real())
        {
            return PLAIDML_DATA_FLOAT16;
        }
        if (element_type.is_signed())
        {
            return PLAIDML_DATA_INT16;
        }
        return PLAIDML_DATA_UINT16;

    case 32:
        if (element_type.is_real())
        {
            return PLAIDML_DATA_FLOAT32;
        }
        if (element_type.is_signed())
        {
            return PLAIDML_DATA_INT32;
        }
        return PLAIDML_DATA_UINT32;

    case 64:
        if (element_type.is_real())
        {
            return PLAIDML_DATA_FLOAT64;
        }
        if (element_type.is_signed())
        {
            return PLAIDML_DATA_INT64;
        }
        return PLAIDML_DATA_UINT64;

    default: break;
    }

    throw ngraph::ngraph_error{
        std::string{"The nGraph PlaidML backend doesn't support the requested element type ("} +
        element_type.c_type_string() + ")"};
}

vp::shape<char> ngraph::runtime::plaidml::to_plaidml(std::shared_ptr<vertexai::ctx>& ctx,
                                                     const ngraph::element::Type& element_type,
                                                     const ngraph::Shape& shape,
                                                     ConversionUse use)
{
    vp::shape<char> ps{ctx, to_plaidml(element_type, use)};
    std::ptrdiff_t stride = 1;
    for (auto dit = shape.begin(); dit != shape.end(); ++dit)
    {
        stride *= *dit;
    }
    for (auto dit = shape.begin(); dit != shape.end(); ++dit)
    {
        if (*dit)
        {
            stride /= *dit;
        }
        ps.add_dimension(*dit, stride);
    }
    return ps;
}

std::string ngraph::runtime::plaidml::tile_converter(const std::string& tensor_name,
                                                     vp::datatype dt)
{
    switch (dt)
    {
    case PLAIDML_DATA_BOOLEAN:
        return "as_uint(" + tensor_name + ", 8)"; // N.B. nGraph boolean semantics
    case PLAIDML_DATA_INT8: return "as_int(" + tensor_name + ", 8)";
    case PLAIDML_DATA_INT16: return "as_int(" + tensor_name + ", 16)";
    case PLAIDML_DATA_INT32: return "as_int(" + tensor_name + ", 32)";
    case PLAIDML_DATA_INT64: return "as_int(" + tensor_name + ", 64)";
    case PLAIDML_DATA_UINT8: return "as_uint(" + tensor_name + ", 8)";
    case PLAIDML_DATA_UINT16: return "as_uint(" + tensor_name + ", 16)";
    case PLAIDML_DATA_UINT32: return "as_uint(" + tensor_name + ", 32)";
    case PLAIDML_DATA_UINT64: return "as_uint(" + tensor_name + ", 64)";
    case PLAIDML_DATA_FLOAT16: return "as_float(" + tensor_name + ", 16)";
    case PLAIDML_DATA_FLOAT32: return "as_float(" + tensor_name + ", 32)";
    case PLAIDML_DATA_FLOAT64: return "as_float(" + tensor_name + ", 64)";
    default: throw std::runtime_error{"Unsupported type conversion"};
    }
}

std::string ngraph::runtime::plaidml::tile_converter(const std::string& tensor_name,
                                                     const ngraph::element::Type& element_type)
{
    if (!element_type.bitwidth())
    {
        return tensor_name;
    }
    return tile_converter(tensor_name, to_plaidml(element_type));
}

vp::variable ngraph::runtime::plaidml::plaidml_logical_to_data(vp::variable var, bool debug)
{
    return builder::Function{"logicalToData", debug}
        .add(builder::Input{var, "I"})
        .add(builder::Output{"O"})
        .add(builder::Elementwise{"O", "as_int(I ? 1 : 0, 8)"})
        .finalize();
}
