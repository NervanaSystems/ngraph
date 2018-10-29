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

#pragma once

#include <plaidml/plaidml++.h>

#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            enum class ConversionUse
            {
                FOR_DATA = 0,
                FOR_IO = 1,
            };

            vertexai::plaidml::datatype to_plaidml(const ngraph::element::Type& element_type,
                                                   ConversionUse use = ConversionUse::FOR_DATA);

            vertexai::plaidml::shape<char> to_plaidml(std::shared_ptr<vertexai::ctx>& ctx,
                                                      const ngraph::element::Type& element_type,
                                                      const ngraph::Shape& shape,
                                                      ConversionUse use = ConversionUse::FOR_DATA);

            std::string tile_converter(const std::string& tensor_name,
                                       vertexai::plaidml::datatype dt);

            std::string tile_converter(const std::string& tensor_name,
                                       const ngraph::element::Type& element_type);

            vertexai::plaidml::variable plaidml_logical_to_data(vertexai::plaidml::variable var,
                                                                bool debug);
        }
    }
}
